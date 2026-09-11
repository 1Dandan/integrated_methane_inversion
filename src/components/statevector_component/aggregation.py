#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import yaml
import xarray as xr
import numpy as np
import warnings
import functools
import heapq

from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from src.inversion_scripts.point_sources import get_point_source_coordinates
from src.inversion_scripts.imi_preview import (
    estimate_averaging_kernel,
    map_sensitivities_to_sv,
    load_sensitivities,
    save_sensitivities,
)
from src.inversion_scripts.classify_TROPOMI_obs_to_CSgrids import (
    latlon_to_cartesian,
    build_kdtree,
)

# Always flush prints in batch jobs
print = functools.partial(print, flush=True)


# -----------------------------------------------------------------------------
# Grid helpers
# -----------------------------------------------------------------------------
def precompute_flat_lonlat(config, sv_ds):
    """
    Flatten grid center lon/lat once and reuse everywhere.

    Returns:
        lon_all     1D float : flattened lon (wrapped to [-180, 180] for GCHP)
        lat_all     1D float : flattened lat
        grid_shape  tuple    : original grid shape for reshape
    """
    if config["UseGCHP"]:
        lats = sv_ds["lats"].values
        lons = sv_ds["lons"].values
        grid_shape = lats.shape  # (nf, Ydim, Xdim)
        lat_all = lats.reshape(-1)
        lon_all = lons.reshape(-1).copy()
        lon_all[lon_all > 180] -= 360
    else:
        lat = sv_ds["lat"].values
        lon = sv_ds["lon"].values
        LON, LAT = np.meshgrid(lon, lat)
        grid_shape = LON.shape
        lon_all = LON.reshape(-1)
        lat_all = LAT.reshape(-1)
    return lon_all, lat_all, grid_shape


def estimate_gridstep_xyz_from_roi(lon_flat, lat_flat, roi_idx, sample_n=4000, random_state=0):
    """
    Estimate a typical nearest-neighbor spacing in xyz chord distance (unit sphere).

    Uses a random subsample of ROI cells and k=2 NN (self + nearest neighbor).
    The returned distance is used to scale xyz so that ~1 grid step ≈ 1 unit.
    """
    roi_idx = np.asarray(roi_idx, dtype=np.int64)
    if roi_idx.size < 2:
        return 1.0

    rng = np.random.default_rng(random_state)
    n = int(min(sample_n, roi_idx.size))
    samp = roi_idx if n == roi_idx.size else rng.choice(roi_idx, size=n, replace=False)

    lon = np.asarray(lon_flat[samp], dtype=float).copy()
    lat = np.asarray(lat_flat[samp], dtype=float)

    xyz = latlon_to_cartesian(lat, lon)

    nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn.fit(xyz)
    dists, _ = nn.kneighbors(xyz, return_distance=True)

    step = np.nanmedian(dists[:, 1])
    if not np.isfinite(step) or step <= 0:
        step = 1.0
    return float(step)


# -----------------------------------------------------------------------------
# Native-grid connectivity graph
# -----------------------------------------------------------------------------
def build_grid_neighbor_graph(
    lon_all,
    lat_all,
    valid_idx=None,
    n_neighbors=8,
    distance_factor=1.6,
):
    """
    Build an undirected physical-neighbor graph for valid state-vector cells.

    The nearest-neighbor model is fit on the FULL model grid so that native-grid
    adjacency is preserved, including across cubed-sphere face boundaries.  If
    valid_idx is supplied, only those cells are queried and only valid-to-valid
    edges are retained.  This avoids building connectivity for grid cells that
    never participate in ROI clustering while preserving the same physical
    neighbor definition as a full-grid graph.

    The returned sparse matrix keeps full-grid shape so downstream code can keep
    indexing it using global flattened grid indices.
    """
    lon_all = np.asarray(lon_all, dtype=float)
    lat_all = np.asarray(lat_all, dtype=float)
    n_full = lon_all.size

    if valid_idx is None:
        valid_idx = np.arange(n_full, dtype=np.int64)
    else:
        valid_idx = np.asarray(valid_idx, dtype=np.int64)

    n_valid = int(valid_idx.size)

    if n_valid <= 1:
        return csr_matrix((n_full, n_full), dtype=np.int8)

    # Cartesian coordinates for the full cubed-sphere grid.
    xyz_all = latlon_to_cartesian(lat_all, lon_all)

    # Fit on the full native grid.  Fitting only on ROI cells could create false
    # neighbor relationships across non-ROI gaps.
    k = min(int(n_neighbors) + 1, n_full)
    nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree")
    nn.fit(xyz_all)

    # Query only cells that can participate in clustering.
    dists, inds = nn.kneighbors(xyz_all[valid_idx], return_distance=True)

    # Native-grid spacing is defined relative to the full grid.
    local_step = dists[:, 1].copy()
    valid_step = np.isfinite(local_step) & (local_step > 0)
    if not np.all(valid_step):
        replacement = np.nanmedian(local_step[valid_step])
        if not np.isfinite(replacement) or replacement <= 0:
            replacement = 1.0
        local_step[~valid_step] = replacement

    # Map global flattened grid index -> position in valid_idx.
    # -1 means that the cell is outside the valid ROI set.
    global_to_valid = np.full(n_full, -1, dtype=np.int64)
    global_to_valid[valid_idx] = np.arange(n_valid, dtype=np.int64)

    row_valid = np.repeat(np.arange(n_valid, dtype=np.int64), k - 1)
    neighbor_global = inds[:, 1:].reshape(-1).astype(np.int64)
    neighbor_dist = dists[:, 1:].reshape(-1)
    neighbor_valid = global_to_valid[neighbor_global]

    # Retain only valid-to-valid candidate edges.
    keep = (
        (neighbor_valid >= 0)
        & np.isfinite(neighbor_dist)
        & (neighbor_dist > 0)
    )

    row_valid = row_valid[keep]
    neighbor_valid = neighbor_valid[keep]
    neighbor_global = neighbor_global[keep]
    neighbor_dist = neighbor_dist[keep]

    # Apply the same local native-grid distance criterion as the full-grid graph.
    threshold = float(distance_factor) * np.maximum(
        local_step[row_valid],
        local_step[neighbor_valid],
    )
    keep = neighbor_dist <= threshold

    row_valid = row_valid[keep]
    neighbor_global = neighbor_global[keep]

    row_global = valid_idx[row_valid]

    # Make graph explicitly symmetric while retaining full-grid indexing.
    rr = np.concatenate([row_global, neighbor_global])
    cc = np.concatenate([neighbor_global, row_global])
    data = np.ones(rr.size, dtype=np.int8)

    graph = coo_matrix((data, (rr, cc)), shape=(n_full, n_full)).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    if graph.nnz:
        graph.data[:] = 1

    print(
        "Built native-grid connectivity graph for "
        f"{n_valid} valid ROI cells out of {n_full} total cells, "
        f"{graph.nnz // 2} undirected ROI-to-ROI edges"
    )
    return graph


def build_latlon_neighbor_graph(grid_shape, valid_idx=None, periodic_lon=True):
    """
    Build exact 8-neighbor connectivity for a regular GCClassic lat/lon grid.

    Only valid-to-valid edges are retained. Longitude is periodic for global
    grids and non-periodic for regional grids. The sparse graph keeps full-grid
    flattened indexing so downstream code is shared with GCHP.
    """
    nlat, nlon = map(int, grid_shape)
    n_full = nlat * nlon

    if valid_idx is None:
        valid_mask = np.ones(n_full, dtype=bool)
        n_valid = n_full
    else:
        valid_idx = np.asarray(valid_idx, dtype=np.int64)
        valid_mask = np.zeros(n_full, dtype=bool)
        valid_mask[valid_idx] = True
        n_valid = int(valid_idx.size)

    if n_valid <= 1:
        return csr_matrix((n_full, n_full), dtype=np.int8)

    idx = np.arange(n_full, dtype=np.int64).reshape(nlat, nlon)
    rows = []
    cols = []

    # East-west neighbors.
    if periodic_lon:
        a = idx
        b = np.roll(idx, -1, axis=1)
    else:
        a = idx[:, :-1]
        b = idx[:, 1:]
    rows.append(a.ravel())
    cols.append(b.ravel())

    # North-south neighbors.
    rows.append(idx[:-1, :].ravel())
    cols.append(idx[1:, :].ravel())

    # Diagonals.
    if periodic_lon:
        rows.append(idx[:-1, :].ravel())
        cols.append(np.roll(idx[1:, :], -1, axis=1).ravel())
        rows.append(idx[:-1, :].ravel())
        cols.append(np.roll(idx[1:, :], 1, axis=1).ravel())
    else:
        rows.append(idx[:-1, :-1].ravel())
        cols.append(idx[1:, 1:].ravel())
        rows.append(idx[:-1, 1:].ravel())
        cols.append(idx[1:, :-1].ravel())

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    keep = valid_mask[rows] & valid_mask[cols] & (rows != cols)
    rows = rows[keep]
    cols = cols[keep]

    rr = np.concatenate([rows, cols])
    cc = np.concatenate([cols, rows])
    data = np.ones(rr.size, dtype=np.int8)

    graph = coo_matrix((data, (rr, cc)), shape=(n_full, n_full)).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    if graph.nnz:
        graph.data[:] = 1

    print(
        "Built GCClassic connectivity graph for "
        f"{n_valid} valid ROI cells out of {n_full} total cells, "
        f"{graph.nnz // 2} undirected ROI-to-ROI edges"
    )
    return graph


# -----------------------------------------------------------------------------
# Country mask sampling
# -----------------------------------------------------------------------------
def sample_country_mask_nearest(lat, lon, mask, res_deg=0.1):
    """Nearest-neighbor sampling on a regular lat/lon mask."""
    lat = np.asarray(lat)
    lon = np.asarray(lon).copy()
    lon[lon > 180] -= 360

    j = np.rint((lat - (-90 + res_deg / 2)) / res_deg).astype(np.int64)
    i = np.rint((lon - (-180 + res_deg / 2)) / res_deg).astype(np.int64)

    j = np.clip(j, 0, mask.shape[0] - 1)
    i = np.clip(i, 0, mask.shape[1] - 1)
    return mask[j, i]


def sample_country_mask_majority(lat, lon, mask, ocean_id=-1, res_deg=0.1):
    """
    Majority vote among 5 samples (center + 4 nudges).
    If any land exists, ocean votes are ignored.
    """
    nudge = 0.8 * res_deg
    s0 = sample_country_mask_nearest(lat, lon, mask, res_deg)
    s1 = sample_country_mask_nearest(lat + nudge, lon, mask, res_deg)
    s2 = sample_country_mask_nearest(lat - nudge, lon, mask, res_deg)
    s3 = sample_country_mask_nearest(lat, lon + nudge, mask, res_deg)
    s4 = sample_country_mask_nearest(lat, lon - nudge, mask, res_deg)

    samples = np.stack([s0, s1, s2, s3, s4], axis=0)
    is_land = samples != ocean_id
    has_land = is_land.any(axis=0)

    eq = samples[:, None, :] == samples[None, :, :]
    eq &= is_land[:, None, :]
    eq &= is_land[None, :, :]
    counts = eq.sum(axis=1)

    best = counts.argmax(axis=0)
    out = samples[best, np.arange(samples.shape[1])]
    out[~has_land] = ocean_id
    return out


def assign_country_ids_valid(
    config,
    sv_ds,
    valid_indices,
    country_mask_ds,
    mask_var="country_id",
    ocean_id=-1,
    res_deg=0.1,
    majority_vote=True,
    lats_flat=None,
    lons_flat=None,
):
    """
    Assign country_id for selected grid cells.

    valid_indices can be a boolean mask on flattened grid or flattened indices.
    """
    valid_indices = np.asarray(valid_indices)
    idx = np.flatnonzero(valid_indices) if valid_indices.dtype == bool else valid_indices.astype(np.int64)

    mask = country_mask_ds[mask_var].values

    if (lats_flat is None) or (lons_flat is None):
        if config["UseGCHP"]:
            lats_flat = sv_ds["lats"].values.reshape(-1)
            lons_flat = sv_ds["lons"].values.reshape(-1)
        else:
            lon = sv_ds["lon"].values
            lat = sv_ds["lat"].values
            lons, lats = np.meshgrid(lon, lat)
            lats_flat = lats.reshape(-1)
            lons_flat = lons.reshape(-1)

    lat_v = lats_flat[idx]
    lon_v = lons_flat[idx]

    if majority_vote:
        return sample_country_mask_majority(lat_v, lon_v, mask, ocean_id=ocean_id, res_deg=res_deg)
    return sample_country_mask_nearest(lat_v, lon_v, mask, res_deg=res_deg)


# -----------------------------------------------------------------------------
# Grouped KMeans utilities (exact total cluster count across groups)
# -----------------------------------------------------------------------------
def allocate_k_per_group(
    group_ids,
    total_k,
    min_k=1,
    max_cluster_size=None,
):
    """
    Allocate exactly total_k clusters across groups.

    If max_cluster_size is supplied, each group receives at least
    ceil(group_size / max_cluster_size) clusters. This is necessary when groups
    are disconnected from each other because clusters cannot merge across them.
    """
    group_ids = np.asarray(group_ids)
    ids, counts = np.unique(group_ids, return_counts=True)
    counts = counts.astype(np.int64)
    total_k = int(total_k)

    if total_k <= 0:
        raise ValueError("total_k must be positive")
    if total_k > int(counts.sum()):
        raise ValueError(
            f"total_k={total_k} exceeds number of points={int(counts.sum())}"
        )

    floor_k = np.full(ids.size, int(min_k), dtype=np.int64)
    if max_cluster_size is not None:
        cap = int(max_cluster_size)
        floor_k = np.maximum(floor_k, (counts + cap - 1) // cap)

    floor_k = np.minimum(floor_k, counts)
    floor_total = int(floor_k.sum())
    if floor_total > total_k:
        raise RuntimeError(
            f"Need at least {floor_total} clusters to preserve connected/grouped "
            f"regions and MaxClusterSize, but only {total_k} were requested."
        )

    raw = counts / counts.sum() * total_k
    k = np.floor(raw).astype(np.int64)
    k = np.maximum(k, floor_k)
    k = np.minimum(k, counts)

    diff = int(total_k - k.sum())
    remainder = raw - np.floor(raw)

    while diff > 0:
        can = np.where(k < counts)[0]
        if can.size == 0:
            raise RuntimeError("Unable to allocate all requested clusters across groups")
        # Favor the group most under its proportional allocation.
        score = raw[can] - k[can]
        j = can[np.argmax(score + 1e-12 * remainder[can])]
        k[j] += 1
        diff -= 1

    while diff < 0:
        can = np.where(k > floor_k)[0]
        if can.size == 0:
            raise RuntimeError("Unable to reduce grouped allocation to requested total")
        score = k[can] - raw[can]
        j = can[np.argmax(score)]
        k[j] -= 1
        diff += 1

    assert int(k.sum()) == total_k
    assert np.all(k >= floor_k)
    assert np.all(k <= counts)
    return ids, k.astype(int)


def _fit_kmeans_exact(features, n_clusters, mini_batch=True, random_state=0, context=""):
    """Fit exactly n_clusters occupied clusters.

    MiniBatchKMeans is useful when K is small relative to N. When K is already
    a large fraction of N, ordinary KMeans is used directly because MiniBatch
    frequently leaves empty centers in that regime. Otherwise MiniBatch is tried
    first and ordinary KMeans is retained as an exact-occupancy fallback.
    """
    X = np.asarray(features)
    n = int(X.shape[0])
    k = int(n_clusters)

    if k <= 0 or k > n:
        raise ValueError(f"Invalid n_clusters={k} for n_samples={n}")
    if k == 1:
        return np.zeros(n, dtype=np.int64)
    if k == n:
        return np.arange(n, dtype=np.int64)

    # Internal implementation choice, not a user-facing configuration parameter.
    # For high K/N, standard KMeans is usually both more reliable and reasonable
    # because each cluster contains only a few points.
    use_minibatch = bool(mini_batch and (k / n < 0.20))
    where = f" ({context})" if context else ""

    if use_minibatch:
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
        )
        labels = km.fit_predict(X).astype(np.int64)
        n_found = int(np.unique(labels).size)
        if n_found == k:
            return labels

        print(
            f"MiniBatchKMeans requested {k} clusters for {n} cells "
            f"(K/N={k/n:.3f}){where} but produced only {n_found} occupied "
            "clusters; retrying this partition with standard KMeans."
        )
    elif mini_batch:
        print(
            f"Using standard KMeans directly for {k} clusters and {n} cells "
            f"(K/N={k/n:.3f}){where}."
        )

    km = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=1,
    )
    labels = km.fit_predict(X).astype(np.int64)
    n_found = int(np.unique(labels).size)

    if n_found != k:
        n_unique = int(np.unique(X, axis=0).shape[0])
        raise RuntimeError(
            f"Standard KMeans requested {k} clusters{where} but produced "
            f"{n_found} occupied clusters. The partition contains {n_unique} "
            f"unique feature vectors among {n} samples."
        )
    return labels


def kmeans_by_group(
    features,
    group_ids,
    num_clusters,
    mini_batch=True,
    random_state=0,
    min_k=1,
    max_cluster_size=None,
):
    """Run exact-K KMeans independently within each connected group.

    Returns labels plus the per-group target allocation so an exact final
    connectivity merge can preserve each disconnected group's required count.
    """
    features = np.asarray(features)
    group_ids = np.asarray(group_ids)

    ids, k_per = allocate_k_per_group(
        group_ids,
        num_clusters,
        min_k=min_k,
        max_cluster_size=max_cluster_size,
    )

    labels = np.full(group_ids.shape[0], -1, dtype=np.int64)
    next_label = 0

    for gid, k_i in zip(ids, k_per):
        idx = np.where(group_ids == gid)[0]
        n = int(idx.size)
        k_i = int(k_i)
        if n == 0:
            continue

        if k_i == 1:
            labels[idx] = next_label
            next_label += 1
            continue
        if k_i == n:
            labels[idx] = np.arange(next_label, next_label + n, dtype=np.int64)
            next_label += n
            continue

        local = _fit_kmeans_exact(
            features[idx],
            k_i,
            mini_batch=mini_batch,
            random_state=random_state,
            context=f"group {gid}",
        )
        labels[idx] = local + next_label
        next_label += k_i

    if next_label != int(num_clusters):
        raise RuntimeError(
            f"Requested {num_clusters} grouped clusters, but generated {next_label}."
        )
    if np.any(labels < 0):
        raise RuntimeError("Grouped KMeans left some selected cells unassigned")

    return labels, ids, k_per


# -----------------------------------------------------------------------------
# Cluster-size cap: split oversized clusters
# -----------------------------------------------------------------------------
def split_oversized_clusters(features, labels, max_cluster_size, mini_batch=True, random_state=0):
    """
    Split clusters with size > max_cluster_size by reclustering within that cluster.

    This may temporarily increase the number of local cluster IDs.  The caller
    subsequently merges adjacent connected pieces back to the exact requested
    cluster count while respecting max_cluster_size.
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    if labels.size == 0:
        return labels

    cap = int(max_cluster_size)
    if cap <= 0:
        return labels

    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    if counts.size == 0 or int(counts.max()) <= cap:
        return labels

    next_label = int(labels.max()) + 1

    while True:
        counts = np.bincount(labels, minlength=int(labels.max()) + 1)
        oversized = np.where(counts > cap)[0]
        if oversized.size == 0:
            break

        c = int(oversized[0])
        idx = np.where(labels == c)[0]
        n = int(idx.size)

        k_split = int((n + cap - 1) // cap)
        if k_split <= 1:
            raise RuntimeError("Internal error while splitting oversized cluster")

        if k_split == n:
            sub = np.arange(n, dtype=int)
        else:
            sub = _fit_kmeans_exact(
                features[idx],
                k_split,
                mini_batch=mini_batch,
                random_state=random_state,
                context=f"oversized cluster {c}",
            )

        # Keep sub==0 as the original label; assign new IDs to remaining pieces.
        for s in range(1, int(np.max(sub)) + 1):
            sel = idx[sub == s]
            if sel.size == 0:
                continue
            labels[sel] = next_label
            next_label += 1

    return labels


# -----------------------------------------------------------------------------
# Connectivity enforcement that preserves the requested final cluster count
# -----------------------------------------------------------------------------
def split_disconnected_components_graph(labels, adjacency):
    """
    Split every disconnected piece of each current label into its own temporary
    label using the native-grid adjacency graph.

    The returned labels are contiguous and 0-based.  This step can increase the
    temporary number of labels; merge_connected_clusters_to_target() then merges
    adjacent connected pieces back to the requested count.
    """
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.size
    if n == 0:
        return labels

    coo = adjacency.tocoo()
    upper = coo.row < coo.col
    r = coo.row[upper]
    c = coo.col[upper]

    same = labels[r] == labels[c]
    r = r[same]
    c = c[same]

    if r.size == 0:
        return np.arange(n, dtype=np.int64)

    rr = np.concatenate([r, c])
    cc = np.concatenate([c, r])
    data = np.ones(rr.size, dtype=np.int8)
    same_label_graph = coo_matrix((data, (rr, cc)), shape=(n, n)).tocsr()

    _, component_ids = connected_components(
        same_label_graph,
        directed=False,
        return_labels=True,
    )
    return component_ids.astype(np.int64)


def connected_partition_groups(adjacency, country_id=None):
    """
    Return connected grouping IDs for the current clustering subset.

    Without country grouping, these are simply spatial connected components.
    With country grouping, graph edges crossing country IDs are removed first, so
    each returned group is both spatially connected and contained in one country.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)

    if country_id is None:
        _, group_ids = connected_components(
            adjacency, directed=False, return_labels=True
        )
        return group_ids.astype(np.int64)

    country_id = np.asarray(country_id)
    coo = adjacency.tocoo()
    same_country = country_id[coo.row] == country_id[coo.col]
    rr = coo.row[same_country]
    cc = coo.col[same_country]
    data = np.ones(rr.size, dtype=np.int8)
    grouped_graph = coo_matrix((data, (rr, cc)), shape=(n, n)).tocsr()
    _, group_ids = connected_components(
        grouped_graph, directed=False, return_labels=True
    )
    return group_ids.astype(np.int64)


def merge_connected_clusters_to_target(
    labels,
    features,
    adjacency,
    target_k,
    max_cluster_size,
    point_group_ids=None,
):
    """
    Merge adjacent connected clusters until exactly target_k remain.

    Each merge is allowed only when:
      - the two clusters share at least one native-grid edge;
      - they belong to the same optional country/group ID;
      - the merged size is <= max_cluster_size.

    Therefore connectivity and MaxClusterSize are preserved.
    """
    labels = np.asarray(labels, dtype=np.int64)
    features = np.asarray(features, dtype=float)
    target_k = int(target_k)
    cap = int(max_cluster_size)

    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.int64)

    n = labels.size
    n_clusters = int(labels.max()) + 1 if n else 0

    if n_clusters < target_k:
        raise RuntimeError(
            f"Only {n_clusters} temporary clusters exist, fewer than target_k={target_k}."
        )
    if n_clusters == target_k:
        sizes = np.bincount(labels, minlength=target_k)
        if (sizes.max() if sizes.size else 0) > cap:
            raise RuntimeError("MaxClusterSize violation before connectivity merge")
        return labels

    sizes = np.bincount(labels, minlength=n_clusters).astype(np.int64)
    if (sizes.max() if sizes.size else 0) > cap:
        raise RuntimeError("Temporary cluster larger than MaxClusterSize before merge")

    feature_sums = np.zeros((n_clusters, features.shape[1]), dtype=float)
    np.add.at(feature_sums, labels, features)

    if point_group_ids is None:
        cluster_group = np.zeros(n_clusters, dtype=np.int64)
    else:
        point_group_ids = np.asarray(point_group_ids)
        first = np.full(n_clusters, n, dtype=np.int64)
        np.minimum.at(first, labels, np.arange(n, dtype=np.int64))
        cluster_group = point_group_ids[first]
        if np.any(point_group_ids != cluster_group[labels]):
            raise RuntimeError("A temporary cluster spans more than one grouping region")

    # Cluster-level adjacency.
    coo = adjacency.tocoo()
    upper = coo.row < coo.col
    r = coo.row[upper]
    c = coo.col[upper]

    a = labels[r]
    b = labels[c]
    diff = a != b
    a = a[diff]
    b = b[diff]

    if a.size:
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        pairs = np.unique(np.column_stack([lo, hi]), axis=0)
    else:
        pairs = np.empty((0, 2), dtype=np.int64)

    parent = np.arange(n_clusters, dtype=np.int64)
    neighbors = [set() for _ in range(n_clusters)]

    for aa, bb in pairs:
        aa = int(aa)
        bb = int(bb)
        if cluster_group[aa] != cluster_group[bb]:
            continue
        neighbors[aa].add(bb)
        neighbors[bb].add(aa)

    def find(x):
        x = int(x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    heap = []

    def push_candidate(a0, b0):
        ra = find(a0)
        rb = find(b0)
        if ra == rb:
            return
        if rb not in neighbors[ra]:
            return
        if cluster_group[ra] != cluster_group[rb]:
            return

        combined = int(sizes[ra] + sizes[rb])
        if combined > cap:
            return

        ca = feature_sums[ra] / sizes[ra]
        cb = feature_sums[rb] / sizes[rb]
        dist2 = float(np.sum((ca - cb) ** 2))
        x, y = (ra, rb) if ra < rb else (rb, ra)

        # Prefer merging small adjacent pieces; use centroid distance as tie-breaker.
        heapq.heappush(heap, (combined, dist2, x, y))

    for aa, bb in pairs:
        push_candidate(int(aa), int(bb))

    active_count = n_clusters

    while active_count > target_k:
        if not heap:
            raise RuntimeError(
                "Unable to merge connected clusters down to the requested count "
                f"without violating MaxClusterSize={cap}. "
                "This can occur if the remaining connected/country regions do not "
                "permit enough legal merges."
            )

        _, _, aa, bb = heapq.heappop(heap)
        ra = find(aa)
        rb = find(bb)

        if ra == rb:
            continue
        if rb not in neighbors[ra]:
            continue
        if cluster_group[ra] != cluster_group[rb]:
            continue
        if sizes[ra] + sizes[rb] > cap:
            continue

        # Keep the larger root to reduce parent-chain depth.
        if sizes[rb] > sizes[ra]:
            ra, rb = rb, ra

        old_neighbors = neighbors[ra] | neighbors[rb]

        parent[rb] = ra
        sizes[ra] += sizes[rb]
        feature_sums[ra] += feature_sums[rb]
        sizes[rb] = 0

        new_neighbors = set()
        for nb in old_neighbors:
            rnb = find(nb)
            if rnb != ra:
                new_neighbors.add(rnb)

        neighbors[ra] = new_neighbors
        neighbors[rb] = set()

        # Replace ra/rb by the surviving root in all adjacent neighbor sets.
        for nb in list(new_neighbors):
            cleaned = {find(x) for x in neighbors[nb]}
            cleaned.discard(nb)
            cleaned.discard(rb)
            cleaned.discard(ra)
            cleaned.add(ra)
            neighbors[nb] = cleaned

        active_count -= 1

        for nb in list(new_neighbors):
            push_candidate(ra, nb)

    root_for_cluster = np.array([find(i) for i in range(n_clusters)], dtype=np.int64)
    point_roots = root_for_cluster[labels]
    _, out = np.unique(point_roots, return_inverse=True)
    out = out.astype(np.int64)

    final_sizes = np.bincount(out, minlength=target_k)
    if np.unique(out).size != target_k:
        raise RuntimeError(
            f"Connectivity merge produced {np.unique(out).size} clusters, expected {target_k}."
        )
    if (final_sizes.max() if final_sizes.size else 0) > cap:
        raise RuntimeError(
            f"Connectivity merge violated MaxClusterSize={cap}; max={final_sizes.max()}."
        )

    return out


def merge_connected_clusters_by_group(
    labels,
    features,
    adjacency,
    point_group_ids,
    group_ids,
    target_k_per_group,
    max_cluster_size,
):
    """Merge temporary connected pieces back to exact targets group-by-group.

    Doing this independently for each spatially connected country/domain group
    prevents the global greedy merge from consuming too many merges in one group
    and later becoming unable to reach the requested total in another group.
    """
    labels = np.asarray(labels, dtype=np.int64)
    features = np.asarray(features, dtype=float)
    point_group_ids = np.asarray(point_group_ids)

    out = np.full(labels.size, -1, dtype=np.int64)
    next_label = 0

    for gid, target_k in zip(group_ids, target_k_per_group):
        pidx = np.where(point_group_ids == gid)[0]
        if pidx.size == 0:
            continue

        local_adj = adjacency[pidx][:, pidx].tocsr()
        local_labels = labels[pidx]
        n_temp = int(np.unique(local_labels).size)
        target_k = int(target_k)

        if n_temp < target_k:
            raise RuntimeError(
                f"Connected group {gid} has only {n_temp} temporary clusters, "
                f"fewer than its target {target_k}."
            )

        try:
            local_out = merge_connected_clusters_to_target(
                local_labels,
                features[pidx],
                local_adj,
                target_k=target_k,
                max_cluster_size=max_cluster_size,
                point_group_ids=None,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Connectivity merge failed in connected group {gid}: "
                f"{pidx.size} cells, {n_temp} temporary clusters, "
                f"target={target_k}, MaxClusterSize={max_cluster_size}. "
                f"Original error: {exc}"
            ) from exc

        out[pidx] = local_out + next_label
        next_label += target_k

    if np.any(out < 0):
        raise RuntimeError("Group-wise connectivity merge left cells unassigned")
    if next_label != int(np.sum(target_k_per_group)):
        raise RuntimeError(
            f"Group-wise connectivity merge produced {next_label} labels, "
            f"expected {int(np.sum(target_k_per_group))}."
        )
    return out


# -----------------------------------------------------------------------------
# Subset clustering: spatial geometry + DOFS-scaled sensitivity
# -----------------------------------------------------------------------------
def cluster_data_kmeans(
    config,
    sensi_flat,
    sv_ds,
    num_clusters,
    mini_batch=False,
    cluster_by_country=False,
    _country_mask_cache=None,
    lon_all=None,
    lat_all=None,
    subset_idx=None,
    grid_shape=None,
    max_cluster_size=None,
    gridstep_xyz=None,
    neighbor_graph=None,
    exact_k=False,
):
    """
    Cluster subset_idx cells into exactly num_clusters connected clusters.

    Output:
        full-grid labels (0 outside subset_idx), labels are 1-based.

    Feature design is unchanged from the original code:
      - geometry: xyz chord distance scaled so ~1 grid step ≈ 1 unit
      - sensitivity: log-compressed sensitivity scaled to ~O(1)

    Necessary fixes:
      - NaN sensitivities are treated as zero information rather than dropping cells.
      - MaxClusterSize is a hard cap.
      - connectivity is enforced without changing the requested final cluster count.
    """
    if subset_idx is None:
        raise ValueError("subset_idx must be provided for clustering")
    if lon_all is None or lat_all is None or grid_shape is None:
        raise ValueError("lon_all, lat_all, and grid_shape must be provided")
    if neighbor_graph is None:
        raise ValueError("neighbor_graph must be provided")

    subset_idx = np.asarray(subset_idx, dtype=np.int64)
    out = np.zeros(sensi_flat.size, dtype=np.int32)

    n_sel = int(subset_idx.size)
    K = int(num_clusters)

    if n_sel == 0 or K <= 0:
        return out.reshape(grid_shape)
    if K > n_sel:
        raise ValueError(f"Cannot create {K} clusters from only {n_sel} selected cells")

    cap = int(max_cluster_size) if max_cluster_size is not None else n_sel
    min_k_for_cap = int((n_sel + cap - 1) // cap)
    if K < min_k_for_cap:
        raise RuntimeError(
            f"{n_sel} cells require at least {min_k_for_cap} clusters to satisfy "
            f"MaxClusterSize={cap}, but only {K} were requested."
        )

    # Do not drop NaNs: a missing sensitivity means zero information for clustering.
    idx = subset_idx
    Z = np.asarray(sensi_flat[idx], dtype=float)
    Z = np.where(np.isfinite(Z), Z, 0.0)

    lon = np.asarray(lon_all[idx], dtype=float)
    lat = np.asarray(lat_all[idx], dtype=float)

    xyz = latlon_to_cartesian(lat, lon)
    step = float(gridstep_xyz) if gridstep_xyz is not None else 1.0
    if (not np.isfinite(step)) or step <= 0:
        step = 1.0
    xyz_feat = xyz / step

    thr = float(config.get("ClusteringThreshold", 1.0))
    thr = max(thr, 1e-12)

    z_raw = np.log1p(np.maximum(Z, 0.0) / thr)
    z_scale = np.nanmedian(z_raw)
    if (not np.isfinite(z_scale)) or z_scale <= 0:
        z_scale = 1.0
    s_feat = z_raw / z_scale

    features = np.column_stack((xyz_feat, s_feat))

    # Native-grid adjacency induced on the selected subset.
    subgraph = neighbor_graph[idx][:, idx].tocsr()

    country_id = None
    if cluster_by_country:
        if _country_mask_cache is not None and "ds" in _country_mask_cache:
            country_mask_ds = _country_mask_cache["ds"]
        else:
            country_mask_ds = xr.open_dataset(config["CountryMaskPath"])
            if _country_mask_cache is not None:
                _country_mask_cache["ds"] = country_mask_ds

        country_id = assign_country_ids_valid(
            config,
            sv_ds,
            idx,
            country_mask_ds,
            lats_flat=lat_all,
            lons_flat=lon_all,
        )

    # Partition the subset into groups that are truly spatially connected. If
    # GroupByCountry is enabled, country boundaries are also respected. This is
    # crucial: a single country ID can contain disconnected islands/ROI pieces.
    partition_group_id = connected_partition_groups(subgraph, country_id)

    # Candidate rounds only need a connected partition to rank by information
    # content; K is a suggestion, not a requirement. Raise it to the feasible
    # per-group floor rather than failing the whole run.
    if not exact_k:
        _, _group_counts = np.unique(partition_group_id, return_counts=True)
        floor_total = int(np.maximum(1, -(-_group_counts // cap)).sum())
        if K < floor_total:
            print(
                f"Candidate partition: requested K={K} is below the feasible "
                f"floor of {floor_total} for this subset; using {floor_total}."
            )
            K = min(floor_total, n_sel)

    # Allocate K across connected groups while reserving enough labels in every
    # group to satisfy MaxClusterSize.
    if K == n_sel:
        labels0 = np.arange(n_sel, dtype=np.int64)
        group_ids, target_k_per_group = allocate_k_per_group(
            partition_group_id,
            K,
            min_k=1,
            max_cluster_size=cap,
        )
    else:
        labels0, group_ids, target_k_per_group = kmeans_by_group(
            features,
            partition_group_id,
            K,
            mini_batch=mini_batch,
            random_state=0,
            min_k=1,
            max_cluster_size=cap,
        )

    # Enforce hard MaxClusterSize, then true spatial connectivity.
    labels0 = split_oversized_clusters(
        features,
        labels0,
        cap,
        mini_batch=mini_batch,
        random_state=0,
    )
    n_after_size = np.unique(labels0).size

    labels0 = split_disconnected_components_graph(labels0, subgraph)
    n_after_connectivity = np.unique(labels0).size

    # Intermediate aggregation levels only need a connected candidate partition;
    # they do NOT need to be merged back to the approximate requested K. Doing
    # that was the source of the failure around agg_level=27. Exact K matters
    # only for the final fill.
    if exact_k:
        labels0 = merge_connected_clusters_by_group(
            labels0,
            features,
            subgraph,
            point_group_ids=partition_group_id,
            group_ids=group_ids,
            target_k_per_group=target_k_per_group,
            max_cluster_size=cap,
        )

    final_k = int(np.unique(labels0).size)
    counts = np.bincount(labels0)
    max_size = int(counts.max()) if counts.size else 0
    if max_size > cap:
        raise RuntimeError(
            f"MaxClusterSize violation after clustering: max={max_size}, cap={cap}."
        )

    mode = "exact final" if exact_k else "candidate"
    print(
        f"KMeans partition ({mode}): requested={K}, after size split={n_after_size}, "
        f"after connectivity split={n_after_connectivity}, final={final_k}, "
        f"max size={max_size}"
    )

    out[idx] = labels0.astype(np.int32) + 1
    return out.reshape(grid_shape)


# -----------------------------------------------------------------------------
# Rank clusters by summed sensitivity and apply threshold
# -----------------------------------------------------------------------------
def get_highest_labels_threshold(labels, sensitivities, threshold):
    """Return cluster IDs whose total sensitivity meets threshold (descending)."""
    lab = np.asarray(labels).ravel()
    sen = np.asarray(sensitivities).ravel()

    # Keep every labeled grid cell. Missing sensitivity = zero information.
    m = lab >= 1
    if not np.any(m):
        return (
            np.array([], dtype=int),
            0,
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    lab_m = lab[m].astype(np.int64)
    sen_m = np.asarray(sen[m], dtype=np.float64)
    sen_m = np.where(np.isfinite(sen_m), sen_m, 0.0)

    max_label = int(lab_m.max())
    total_sensi = np.bincount(lab_m, weights=sen_m, minlength=max_label + 1)
    counts = np.bincount(lab_m, minlength=max_label + 1)

    ids = np.arange(1, max_label + 1, dtype=np.int64)
    order = np.argsort(total_sensi[1:])[::-1]

    n_clusters = ids[order]
    n_sensis = total_sensi[1:][order]

    keep = n_sensis >= threshold
    n = int(np.count_nonzero(keep))
    return (
        n_clusters[:n],
        n,
        counts[1:][order][:n],
        np.round(n_sensis[:n], 2),
    )


# -----------------------------------------------------------------------------
# Max cluster size presets (or config override)
# -----------------------------------------------------------------------------
def get_max_cluster_size(config, sensitivities, desired_element_num):
    """Choose MaxClusterSize from presets unless overridden by config."""
    if config["UseGCHP"]:
        preset = {
            "720": 128,
            "360": 64,
            "180": 32,
            "90": 16,
            "48": 8,
            "24": 4,
        }.get(str(config.get("CS_RES")), 64)
    else:
        preset = {
            "0.125x0.15625": 128,
            "0.25x0.3125": 64,
            "0.5x0.625": 32,
            "2.0x2.5": 16,
            "4.0x5.0": 8,
        }.get(str(config.get("Res")), 64)

    max_cluster_size = (
        int(config["MaxClusterSize"])
        if "MaxClusterSize" in config
        else int(preset)
    )

    # Feasibility check: exact requested cluster count must be sufficient to
    # cover all native ROI elements under the hard size cap.
    background_elements_needed = int(
        np.ceil(len(sensitivities) / max_cluster_size)
    )
    if background_elements_needed > desired_element_num:
        raise Exception(
            "Error: too few clusters to satisfy MaxClusterSize.\n"
            + f"At least {background_elements_needed} clusters are needed for "
            + f"{len(sensitivities)} native elements with MaxClusterSize={max_cluster_size}.\n"
            + "Increase NumberOfElements or MaxClusterSize."
        )

    print(f"MaxClusterSize set to: {max_cluster_size} elements in a cluster")
    return max_cluster_size


# -----------------------------------------------------------------------------
# Force native-resolution elements at point-source locations
# -----------------------------------------------------------------------------
def force_native_res_pixels(config, clusters_ds, sensitivities):
    """Snap point sources to grid cells and raise their sensitivities above threshold."""
    dofs_max = float(config["ClusteringThreshold"]) + 0.1 if "ClusteringThreshold" in config else 1.1

    raw_coords = get_point_source_coordinates(config)
    if len(raw_coords) == 0:
        print("No ForcedNativeResolutionElements or PointSourceDatasets specified in config file.")
        return sensitivities

    if config["UseGCHP"]:
        grid_lats = clusters_ds["lats"].values
        grid_lons = clusters_ds["lons"].values
        lon_min, lon_max = -180, 180
        lat_min, lat_max = -90, 90
    else:
        lat = clusters_ds["lat"].values
        lon = clusters_ds["lon"].values
        grid_lons, grid_lats = np.meshgrid(lon, lat)
        delta_lon = float(np.median(np.abs(np.diff(lon))))
        delta_lat = float(np.median(np.abs(np.diff(lat))))
        lon_min = max(float(np.nanmin(grid_lons) - delta_lon / 2), -180.0)
        lon_max = min(float(np.nanmax(grid_lons) + delta_lon / 2), 180.0)
        lat_min = max(float(np.nanmin(grid_lats) - delta_lat / 2), -90.0)
        lat_max = min(float(np.nanmax(grid_lats) + delta_lat / 2), 90.0)

    pts = np.asarray(raw_coords, dtype=float)
    pts_lat = pts[:, 0]
    pts_lon = pts[:, 1].copy()
    pts_lon[pts_lon > 180] -= 360

    roi_mask = (
        (pts_lat >= lat_min)
        & (pts_lat <= lat_max)
        & (pts_lon >= lon_min)
        & (pts_lon <= lon_max)
    )
    raw_coords = np.stack([pts_lat[roi_mask], pts_lon[roi_mask]], axis=1).tolist()

    if len(raw_coords) == 0:
        print("No forced point sources found in the region of interest.")
        return sensitivities

    print(f"Found {len(raw_coords)} point sources at the grid resolution in the region of interest.")

    kdtree, _ = build_kdtree(grid_lats, grid_lons)
    pts = np.asarray(raw_coords, dtype=float)
    q_cart = latlon_to_cartesian(pts[:, 0], pts[:, 1])

    _, idx_flat = kdtree.query(q_cart, k=1)
    idx_flat = np.unique(np.asarray(idx_flat).reshape(-1))

    print(f"{len(raw_coords)} sources → {len(idx_flat)} grid cells")

    if "NumberOfElements" in config:
        max_n = int(config["NumberOfElements"])
        if len(idx_flat) > max_n:
            idx_flat = idx_flat[:max_n]

    clusters_flat = clusters_ds["StateVector"].values.reshape(-1)
    cluster_ids = clusters_flat[idx_flat]

    valid = np.isfinite(cluster_ids) & (cluster_ids >= 1)
    forced_sv_idx = np.unique(cluster_ids[valid].astype(int) - 1)

    sensitivities[forced_sv_idx] = dofs_max
    return sensitivities


# -----------------------------------------------------------------------------
# Assignment helper
# -----------------------------------------------------------------------------
def assign_selected_local_clusters(
    labels_flat,
    subset_idx,
    out_labels,
    selected_local_labels,
    current_max_label,
):
    """Map selected local cluster IDs to new sequential global ROI labels."""
    selected_local_labels = np.asarray(selected_local_labels, dtype=np.int64)
    if selected_local_labels.size == 0:
        return current_max_label, 0

    out_flat_subset = np.asarray(out_labels).reshape(-1)[subset_idx].astype(np.int64, copy=False)
    max_out = int(out_flat_subset.max()) if out_flat_subset.size else 0
    if max_out <= 0:
        raise RuntimeError("No positive local cluster IDs found for assignment")

    lut = np.full(max_out + 1, -1, dtype=np.int32)
    label_start = current_max_label + 1
    new_ids = np.arange(
        label_start,
        label_start + selected_local_labels.size,
        dtype=np.int32,
    )
    lut[selected_local_labels] = new_ids

    valid = (out_flat_subset >= 1) & (out_flat_subset <= max_out)
    mapped = np.full(out_flat_subset.shape, -1, dtype=np.int32)
    mapped[valid] = lut[out_flat_subset[valid]]
    keep = mapped > 0

    labels_flat[subset_idx[keep]] = mapped[keep]
    current_max_label += int(selected_local_labels.size)
    return current_max_label, int(np.count_nonzero(keep))


# -----------------------------------------------------------------------------
# Core aggregation driver (ROI clustering + buffer reattachment)
# -----------------------------------------------------------------------------
def update_sv_clusters(config, flat_sensi, orig_sv_ds):
    """
    Create a reduced state vector.

    Necessary fixes relative to the previous version:
      1. MaxClusterSize is a hard cap on every final ROI cluster.
      2. Missing sensitivities are treated as zero information, not dropped cells.
      3. Connectivity enforcement uses a native-grid graph and preserves the exact
         requested number of clusters.
      4. Candidate selection preserves feasibility of the remaining label budget.
      5. No remaining pixels are ever dumped into the final state-vector label.
    """
    if config["ClusteringMethod"] == "kmeans":
        mini_batch = False
    elif config["ClusteringMethod"] == "mini-batch-kmeans":
        mini_batch = True
    else:
        raise Exception("Error: Invalid Clustering Method. Use 'kmeans' or 'mini-batch-kmeans'.")

    orig_sv = orig_sv_ds["StateVector"]
    orig_sv_np = orig_sv.values

    desired_num_labels = int(config["NumberOfElements"] - config["nBufferClusters"])
    last_ROI_element = int(np.nanmax(orig_sv_np) - config["nBufferClusters"])

    if "ClusteringThreshold" in config:
        dofs_threshold = float(config["ClusteringThreshold"])
    else:
        dofs_threshold = float(np.nansum(flat_sensi) / desired_num_labels)
        if dofs_threshold > 1:
            print(
                f"Estimated dofs per element too high ({dofs_threshold}), "
                "resetting ClusteringThreshold to 1"
            )
            dofs_threshold = 1.0
    print(f"Target DOFS per cluster (ClusteringThreshold): {dofs_threshold}")

    max_cluster_size = get_max_cluster_size(config, flat_sensi, desired_num_labels)

    if "GroupByCountry" in config:
        cluster_by_country = bool(config["GroupByCountry"])
    else:
        warnings.warn('"GroupByCountry" not found in config file. Continuing without clustering by country.')
        cluster_by_country = False

    finite = np.isfinite(orig_sv_np) & (orig_sv_np > 0)
    buffer_threshold = int(np.nanmax(orig_sv_np)) - int(config["nBufferClusters"])

    is_buffer = finite & (orig_sv_np > buffer_threshold)
    is_roi = finite & (orig_sv_np <= last_ROI_element)

    buffer_labels_np = np.where(is_buffer, orig_sv_np, 0.0)
    labels_np = np.where(is_roi, 0.0, np.nan)
    sv_np = np.where(is_roi, orig_sv_np, 0.0)

    sensi_da = map_sensitivities_to_sv(flat_sensi, orig_sv, last_ROI_element)
    sensi_np = sensi_da.values
    print(f"Reducing to {desired_num_labels} elements")

    lon_all, lat_all, grid_shape = precompute_flat_lonlat(config, orig_sv_ds)

    sensi_flat = sensi_np.reshape(-1)
    labels_flat = labels_np.reshape(-1)
    roi_flat_idx = np.flatnonzero(is_roi.reshape(-1))

    n_roi_cells = int(roi_flat_idx.size)
    if desired_num_labels > n_roi_cells:
        raise RuntimeError(
            f"Requested {desired_num_labels} ROI labels for only {n_roi_cells} ROI cells."
        )

    min_labels_required = int((n_roi_cells + max_cluster_size - 1) // max_cluster_size)
    if desired_num_labels < min_labels_required:
        raise RuntimeError(
            f"Need at least {min_labels_required} ROI labels for {n_roi_cells} cells "
            f"with MaxClusterSize={max_cluster_size}; requested {desired_num_labels}."
        )

    print(f"ROI grid cells: {n_roi_cells}")
    print(f"Mean final cluster size if perfectly even: {n_roi_cells / desired_num_labels:.2f}")

    gridstep_xyz = estimate_gridstep_xyz_from_roi(
        lon_all,
        lat_all,
        roi_flat_idx,
        sample_n=4000,
        random_state=0,
    )

    # Build once; every clustering call uses an induced subgraph of this graph.
    # GCHP uses physical xyz neighbors to handle cubed-sphere face boundaries.
    # GCClassic uses exact structured-grid adjacency.
    if config["UseGCHP"]:
        neighbor_graph = build_grid_neighbor_graph(
            lon_all,
            lat_all,
            valid_idx=roi_flat_idx,
        )
    else:
        neighbor_graph = build_latlon_neighbor_graph(
            grid_shape,
            valid_idx=roi_flat_idx,
            periodic_lon=not bool(config.get("isRegional", False)),
        )

    cluster_pairs = np.arange(1, max_cluster_size + 1)
    country_cache = {}
    current_max_label = 0

    # Once successive aggregation levels stop yielding meaningful numbers of
    # clusters, climbing further only burns time: the connected/country floor
    # prevents candidate partitions from coarsening any more. Fall through to
    # the exact final clustering instead.
    stall_levels = 0
    stall_limit = 3
    min_yield_frac = 1e-3
    force_final = False

    for agg_level in cluster_pairs:
        subset_idx = roi_flat_idx[labels_flat[roi_flat_idx] == 0]
        elements_left = int(subset_idx.size)
        if elements_left == 0:
            break

        clusters_left = int(desired_num_labels - current_max_label)
        if clusters_left <= 0:
            raise RuntimeError(
                f"No labels remain, but {elements_left} ROI cells are still unassigned."
            )

        min_clusters_needed = int(
            (elements_left + max_cluster_size - 1) // max_cluster_size
        )
        if clusters_left < min_clusters_needed:
            raise RuntimeError(
                f"Infeasible remaining problem: {elements_left} cells, {clusters_left} labels, "
                f"MaxClusterSize={max_cluster_size}; at least {min_clusters_needed} labels are needed."
            )
        if clusters_left > elements_left:
            raise RuntimeError(
                f"Infeasible remaining problem: {clusters_left} labels for only {elements_left} cells."
            )

        # Final fill occurs when we reach MaxClusterSize or the remaining label
        # budget is already at its minimum feasible value under the hard cap.
        final_fill = (
            agg_level == max_cluster_size
            or clusters_left == min_clusters_needed
            or force_final
        )

        if final_fill:
            print(
                f"Final clustering: assigning {elements_left} remaining cells "
                f"to exactly {clusters_left} clusters."
            )
            out_labels = cluster_data_kmeans(
                config,
                sensi_flat,
                orig_sv_ds,
                clusters_left,
                mini_batch,
                cluster_by_country,
                _country_mask_cache=country_cache,
                lon_all=lon_all,
                lat_all=lat_all,
                subset_idx=subset_idx,
                grid_shape=grid_shape,
                max_cluster_size=max_cluster_size,
                gridstep_xyz=gridstep_xyz,
                neighbor_graph=neighbor_graph,
                exact_k=True,
            )

            out_flat_subset = np.asarray(out_labels).reshape(-1)[subset_idx].astype(np.int64)
            if np.any(out_flat_subset <= 0):
                raise RuntimeError("Final clustering left some ROI cells without a local label")

            local_labels = np.unique(out_flat_subset)
            if local_labels.size != clusters_left:
                raise RuntimeError(
                    f"Final clustering requested {clusters_left} clusters but produced {local_labels.size}."
                )

            current_max_label, n_assigned = assign_selected_local_clusters(
                labels_flat,
                subset_idx,
                out_labels,
                local_labels,
                current_max_label,
            )

            if n_assigned != elements_left:
                raise RuntimeError(
                    f"Final clustering assigned {n_assigned}/{elements_left} remaining cells."
                )
            break

        if agg_level == 1:
            out_labels = sv_np.astype(np.int32)
        else:
            # ceil, not round/floor: candidate cluster count must remain feasible
            # under the current aggregation level.
            n_clusters = int(np.ceil(elements_left / agg_level))
            n_clusters = min(n_clusters, elements_left)

            out_labels = cluster_data_kmeans(
                config,
                sensi_flat,
                orig_sv_ds,
                n_clusters,
                mini_batch,
                cluster_by_country,
                _country_mask_cache=country_cache,
                lon_all=lon_all,
                lat_all=lat_all,
                subset_idx=subset_idx,
                grid_shape=grid_shape,
                max_cluster_size=max_cluster_size,
                gridstep_xyz=gridstep_xyz,
                neighbor_graph=neighbor_graph,
            )

        n_max_labels, _, num_elements, _ = get_highest_labels_threshold(
            out_labels,
            sensi_np,
            dofs_threshold,
        )

        if len(n_max_labels) == 0:
            stall_levels += 1
            if stall_levels >= stall_limit:
                print(
                    f"Aggregation ladder stalled at level {agg_level}; "
                    "proceeding to final clustering."
                )
                force_final = True
            continue

        # Select high-information candidate clusters greedily, but after every
        # accepted cluster preserve BOTH remaining feasibility conditions:
        #   ceil(cells_left / MaxClusterSize) <= labels_left <= cells_left
        # This prevents early singleton clusters from consuming so many labels
        # that the remaining region becomes impossible to cluster legally.
        selected_labels = []
        selected_sizes = []
        cells_remaining = elements_left
        labels_remaining = clusters_left

        for candidate_label, candidate_size in zip(n_max_labels, num_elements):
            candidate_size = int(candidate_size)

            if candidate_size <= 0 or candidate_size > max_cluster_size:
                continue
            if labels_remaining <= 0:
                break

            cells_after = cells_remaining - candidate_size
            labels_after = labels_remaining - 1

            if cells_after < 0 or labels_after < 0:
                continue

            min_needed_after = (
                int((cells_after + max_cluster_size - 1) // max_cluster_size)
                if cells_after > 0
                else 0
            )
            max_possible_after = cells_after  # at least one cell per remaining label

            if labels_after < min_needed_after:
                continue
            if labels_after > max_possible_after:
                continue

            selected_labels.append(int(candidate_label))
            selected_sizes.append(candidate_size)
            cells_remaining = cells_after
            labels_remaining = labels_after

        if len(selected_labels) == 0:
            stall_levels += 1
            if stall_levels >= stall_limit:
                print(
                    f"Aggregation ladder stalled at level {agg_level}; "
                    "proceeding to final clustering."
                )
                force_final = True
            continue

        if len(selected_labels) < max(1, int(min_yield_frac * clusters_left)):
            stall_levels += 1
        else:
            stall_levels = 0
        if stall_levels >= stall_limit:
            print(
                f"Aggregation ladder yield collapsed at level {agg_level}; "
                "proceeding to final clustering."
            )
            force_final = True

        selected_labels = np.asarray(selected_labels, dtype=np.int64)

        print(f"assigning {len(selected_labels)} labels with agg level: {agg_level}")

        current_max_label, n_assigned = assign_selected_local_clusters(
            labels_flat,
            subset_idx,
            out_labels,
            selected_labels,
            current_max_label,
        )

        print(
            f"Assigned {n_assigned} cells; current labels: "
            f"{current_max_label}/{desired_num_labels}"
        )

    # Safety final fill. This replaces the old catch-all assignment to the last
    # label. Every remaining cell must be explicitly clustered.
    subset_idx = roi_flat_idx[labels_flat[roi_flat_idx] == 0]
    elements_left = int(subset_idx.size)
    clusters_left = int(desired_num_labels - current_max_label)

    if elements_left > 0:
        if clusters_left <= 0:
            raise RuntimeError(
                f"Used all {desired_num_labels} labels but {elements_left} ROI cells remain unassigned."
            )

        min_clusters_needed = int(
            (elements_left + max_cluster_size - 1) // max_cluster_size
        )
        if clusters_left < min_clusters_needed or clusters_left > elements_left:
            raise RuntimeError(
                f"Cannot finish aggregation: {elements_left} cells, {clusters_left} labels, "
                f"MaxClusterSize={max_cluster_size}."
            )

        print(
            f"Safety final clustering: {elements_left} cells into "
            f"{clusters_left} clusters."
        )

        out_labels = cluster_data_kmeans(
            config,
            sensi_flat,
            orig_sv_ds,
            clusters_left,
            mini_batch,
            cluster_by_country,
            _country_mask_cache=country_cache,
            lon_all=lon_all,
            lat_all=lat_all,
            subset_idx=subset_idx,
            grid_shape=grid_shape,
            max_cluster_size=max_cluster_size,
            gridstep_xyz=gridstep_xyz,
            neighbor_graph=neighbor_graph,
            exact_k=True,
        )

        out_flat_subset = np.asarray(out_labels).reshape(-1)[subset_idx].astype(np.int64)
        if np.any(out_flat_subset <= 0):
            raise RuntimeError("Safety final clustering left cells without a local label")

        local_labels = np.unique(out_flat_subset)
        if local_labels.size != clusters_left:
            raise RuntimeError(
                f"Safety final clustering requested {clusters_left} clusters but produced {local_labels.size}."
            )

        current_max_label, n_assigned = assign_selected_local_clusters(
            labels_flat,
            subset_idx,
            out_labels,
            local_labels,
            current_max_label,
        )
        if n_assigned != elements_left:
            raise RuntimeError(
                f"Safety final clustering assigned {n_assigned}/{elements_left} cells."
            )

    # No catch-all label is allowed.
    n_unassigned = int(np.count_nonzero(labels_flat[roi_flat_idx] == 0))
    if n_unassigned > 0:
        raise RuntimeError(f"Aggregation ended with {n_unassigned} unassigned ROI cells")

    if current_max_label != desired_num_labels:
        raise RuntimeError(
            f"Expected exactly {desired_num_labels} ROI labels, generated {current_max_label}."
        )

    # Hard final validation of the exact output requested by NumberOfElements and
    # MaxClusterSize.
    roi_labels = labels_flat[roi_flat_idx].astype(np.int64)
    final_ids, final_sizes = np.unique(roi_labels, return_counts=True)

    if final_ids.size != desired_num_labels:
        raise RuntimeError(
            f"Expected {desired_num_labels} final ROI clusters, found {final_ids.size}."
        )
    if final_sizes.max() > max_cluster_size:
        bad = int(np.argmax(final_sizes))
        raise RuntimeError(
            f"MaxClusterSize violation: ROI label {final_ids[bad]} has "
            f"{final_sizes[bad]} cells; maximum allowed is {max_cluster_size}."
        )

    print("\nFinal ROI cluster statistics")
    print("----------------------------")
    print(f"Number of clusters: {final_ids.size}")
    print(f"Minimum cluster size: {final_sizes.min()}")
    print(f"Maximum cluster size: {final_sizes.max()}")
    print(f"Mean cluster size: {final_sizes.mean():.2f}")
    print(f"Median cluster size: {np.median(final_sizes):.1f}")

    # Compress buffer labels to follow reduced ROI range.
    cluster_number_diff = int(last_ROI_element - current_max_label)
    buf = buffer_labels_np.copy()
    buf_mask = buf > 0
    buf[buf_mask] = buf[buf_mask] - cluster_number_diff

    # Merge: buffer where present, otherwise ROI labels.
    statevector_np = np.where(buf_mask, buf, labels_np)

    # Write output dataset.
    refyear = 2000
    fillvalue = -9999
    statevector_np = np.nan_to_num(statevector_np, nan=fillvalue)

    if config["UseGCHP"]:
        da_statevector = xr.DataArray(
            statevector_np[None, ...],
            dims=["time", "nf", "Ydim", "Xdim"],
            coords=dict(
                time=(["time"], [0.0]),
                lats=(["nf", "Ydim", "Xdim"], orig_sv_ds["lats"].values),
                lons=(["nf", "Ydim", "Xdim"], orig_sv_ds["lons"].values),
            ),
            attrs=dict(units="1", missing_value=fillvalue, _FillValue=fillvalue),
        )
        ds_statevector = xr.Dataset({"StateVector": da_statevector})

        ds_statevector.lats.attrs["units"] = "degrees_north"
        ds_statevector.lats.attrs["long_name"] = "Latitude"
        ds_statevector.lons.attrs["units"] = "degrees_east"
        ds_statevector.lons.attrs["long_name"] = "Longitude"
        ds_statevector["time"].attrs = dict(
            units="days since {}-01-01 00:00:00".format(refyear),
            delta_t="0000-01-00 00:00:00",
            axis="T",
            standard_name="Time",
            long_name="Time",
            calendar="standard",
        )

        if "corner_lats" in orig_sv_ds.variables:
            ds_statevector["corner_lats"] = orig_sv_ds["corner_lats"]
        if "corner_lons" in orig_sv_ds.variables:
            ds_statevector["corner_lons"] = orig_sv_ds["corner_lons"]

        if config.get("STRETCH_GRID", False):
            for k in ["STRETCH_FACTOR", "TARGET_LAT", "TARGET_LON"]:
                if k in config:
                    ds_statevector.attrs[k] = np.float32(config[k])

    else:
        da_statevector = xr.DataArray(
            statevector_np,
            dims=orig_sv.dims,
            coords=orig_sv.coords,
            attrs=dict(units="1", missing_value=fillvalue, _FillValue=fillvalue),
        ).expand_dims(time=[0.0])

        ds_statevector = da_statevector.to_dataset(name="StateVector")
        ds_statevector["time"].attrs = dict(
            units="days since {}-01-01 00:00:00".format(refyear),
            delta_t="0000-01-00 00:00:00",
            axis="T",
            standard_name="Time",
            long_name="Time",
            calendar="standard",
        )
        ds_statevector.lat.attrs["units"] = "degrees_north"
        ds_statevector.lat.attrs["long_name"] = "Latitude"
        ds_statevector.lon.attrs["units"] = "degrees_east"
        ds_statevector.lon.attrs["long_name"] = "Longitude"

    return ds_statevector


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
        native_state_vector_path = sys.argv[2]
        state_vector_path = sys.argv[3]
        preview_dir = sys.argv[4]
        tropomi_cache = sys.argv[5]
        kf_index = int(sys.argv[6]) if len(sys.argv) > 6 else None

        config = yaml.load(open(config_path), Loader=yaml.FullLoader)
        original_clusters_ds = xr.open_dataset(native_state_vector_path).squeeze()

        native_labels = original_clusters_ds["StateVector"].squeeze()
        last_ROI_element = int(
            np.nanmax(native_labels.values) - int(config["nBufferClusters"])
        )

        sensitivities = load_sensitivities(
            preview_dir,
            filename="native_sensitivities.nc",
            expected_size=last_ROI_element,
            kf_index=kf_index,
        )

        if sensitivities is None:
            if kf_index is None:
                print(
                    "Native sensitivity cache not found. "
                    "Calculating sensitivities from NativeStateVector.nc."
                )
            else:
                print(
                    f"Native sensitivity cache not found for period {kf_index}. "
                    "Calculating sensitivities from NativeStateVector.nc."
                )
                print(f"Dynamically generating clusters for period: {kf_index}.")

            sensitivity_args = [
                config,
                native_state_vector_path,
                preview_dir,
                tropomi_cache,
                False,
            ]
            if kf_index is not None:
                sensitivity_args.append(kf_index)

            sensitivities = estimate_averaging_kernel(*sensitivity_args)

            save_sensitivities(
                sensitivities,
                preview_dir,
                filename="native_sensitivities.nc",
                kf_index=kf_index,
                config=config,
                state_vector_path=native_state_vector_path,
            )
        else:
            if kf_index is None:
                print(
                    "Using cached native sensitivities; skipping native "
                    "averaging-kernel estimation."
                )
            else:
                print(
                    f"Using cached native sensitivities for period {kf_index}; "
                    "skipping native averaging-kernel estimation."
                )

        sensitivities = force_native_res_pixels(
            config,
            original_clusters_ds,
            sensitivities,
        )

        print(
            "Creating clusters based on information content and spatial proximity.\n"
            "Using ClusteringMethod: 'mini-batch-kmeans' may be faster but less accurate."
        )

        new_sv = update_sv_clusters(config, sensitivities, original_clusters_ds)

        new_sv.to_netcdf(
            state_vector_path,
            encoding={v: {"zlib": True, "complevel": 1} for v in new_sv.data_vars},
        )

        original_clusters_ds.close()

    except Exception as err:
        with open(".aggregation_error.txt", "w") as f:
            f.write(
                "This file is used to tell the controlling script that state vector clustering failed"
            )
        print(err)
        sys.exit(1)