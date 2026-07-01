#!/usr/bin/env bash
set -euo pipefail

file="${1:?Usage: $0 HISTORY.rc}"

old="SpeciesConc"
new="BaseSpeciesConc"

cp -p "$file" "${file}.bak"

tmp="$(mktemp "${file}.tmp.XXXXXX")"
awk_script="$(mktemp)"

trap 'rm -f "$tmp" "$awk_script"' EXIT

cat > "$awk_script" <<'AWK'
function spaces(n) {
    if (n <= 0) {
        return ""
    }
    return sprintf("%" n "s", "")
}

function rename_collection_prefix(line,    ws, rest) {
    # Preserve leading whitespace exactly.
    match(line, /^[ \t]*/)
    ws = substr(line, 1, RLENGTH)
    rest = substr(line, RLENGTH + 1)

    # Rename only the left-hand collection prefix:
    #   SpeciesConc.template -> BaseSpeciesConc.template
    # Do not rename field names such as SpeciesConcVV_CH4.
    if (rest ~ "^" old "\\.") {
        sub("^" old "\\.", new ".", rest)
        line = ws rest
    }

    return line
}

function continuation_indent(line,    qpos) {
    qpos = index(line, "'")
    return spaces(qpos - 1)
}

function field_suffix(line,    q1, rest, q2) {
    # Reuse whatever follows the first quoted field.
    #
    # GCHP style:
    #   'SpeciesConcVV_CH4 ', 'GCHPchem',
    # returns:
    #   , 'GCHPchem',
    #
    # GCClassic style:
    #   'SpeciesConcVV_?ALL? ',
    # returns:
    #   ,
    q1 = index(line, "'")
    if (q1 == 0) {
        return ","
    }

    rest = substr(line, q1 + 1)
    q2 = index(rest, "'")
    if (q2 == 0) {
        return ","
    }

    return substr(rest, q2 + 1)
}

function get_field_width(line,    q1, rest, q2, field) {
    # Keep the same quoted-field width as the original first SpeciesConc field.
    q1 = index(line, "'")
    if (q1 == 0) {
        return 16
    }

    rest = substr(line, q1 + 1)
    q2 = index(rest, "'")
    if (q2 == 0) {
        return 16
    }

    field = substr(rest, 1, q2 - 1)
    return length(field)
}

function pad_field(name, width) {
    return sprintf("%-" width "s", name)
}

function print_base_section(    i, newline, indent, suffix, width, field_prefix, in_fields) {
    if (sep == "") {
        sep = "#============================================================================"
    }

    # Add separator only between SpeciesConc and BaseSpeciesConc.
    print sep

    in_fields = 0

    for (i = 1; i <= n; i++) {
        newline = rename_collection_prefix(block[i])

        # Always print section terminator.
        if (newline ~ /^[[:space:]]*::[[:space:]]*$/) {
            print newline
            in_fields = 0
            continue
        }

        # Replace the entire SpeciesConc.fields block with a clean
        # BaseSpeciesConc.fields block containing only three entries.
        if (newline ~ "^[[:space:]]*" new "\\.fields:") {
            indent = continuation_indent(newline)
            suffix = field_suffix(newline)
            width = get_field_width(newline)

            # Everything before the first quoted field, e.g.
            # "  BaseSpeciesConc.fields:          "
            field_prefix = substr(newline, 1, index(newline, "'") - 1)

            print field_prefix "'" pad_field("SpeciesConcVV_CH4", width) "'" suffix
            print indent       "'" pad_field("Met_AIRDEN",        width) "'" suffix
            print indent       "'" pad_field("Met_BXHEIGHT",      width) "'" suffix

            # Skip all inherited SpeciesConc field continuation lines,
            # such as SpeciesConcVV_CH4_jac0001, jac0002, etc.
            in_fields = 1
            continue
        }

        # Skip continuation lines inside the original SpeciesConc.fields block.
        if (in_fields) {
            continue
        }

        # Copy all other SpeciesConc settings normally.
        print newline
    }
}

{
    line = $0

    # If skipping an old BaseSpeciesConc section, continue until ::.
    if (skip_base_section) {
        if (line ~ /^[[:space:]]*::[[:space:]]*$/) {
            skip_base_section = 0
        }
        next
    }

    # Hold separator lines temporarily so that a separator immediately before
    # an old BaseSpeciesConc section is not kept as an extra duplicate.
    # This does not modify blank lines elsewhere.
    if (line ~ /^#=+/) {
        pending_sep = line
        sep = line
        next
    }

    # Remove any existing old/wrong BaseSpeciesConc section.
    # It will be recreated cleanly.
    if (line ~ "^[[:space:]]*" new "\\.") {
        pending_sep = ""
        skip_base_section = 1
        next
    }

    # Flush pending separator before normal lines.
    if (pending_sep != "") {
        print pending_sep
        pending_sep = ""
    }

    # Enter COLLECTIONS block.
    if (line ~ /^[[:space:]]*COLLECTIONS:/) {
        in_collections = 1
    }

    # Remove existing BaseSpeciesConc entries in COLLECTIONS,
    # commented or uncommented. We will re-add one uncommented entry.
    if (in_collections && line ~ "#?[[:space:]]*'" new "'[[:space:]]*,") {
        next
    }

    print line

    # Add uncommented BaseSpeciesConc right below active SpeciesConc.
    if (in_collections && !added_collection && line ~ "'" old "'[[:space:]]*,") {
        match(line, "'" old "'[[:space:]]*,")
        prefix = substr(line, 1, RSTART - 1)

        # Only add below an active SpeciesConc line, not a commented one.
        if (prefix !~ /#[[:space:]]*$/) {
            indent = spaces(RSTART - 1)
            print indent "'" new "',"
            added_collection = 1
        }
    }

    # Leave COLLECTIONS block.
    if (in_collections && line ~ /^[[:space:]]*::[[:space:]]*$/) {
        in_collections = 0
    }

    # Start collecting SpeciesConc section.
    if (!done_section && line ~ "^[[:space:]]*" old "\\.") {
        in_block = 1
    }

    if (in_block) {
        block[++n] = line
    }

    # End of SpeciesConc section.
    if (in_block && line ~ /^[[:space:]]*::[[:space:]]*$/) {
        print_base_section()

        in_block = 0
        done_section = 1
        n = 0
    }
}

END {
    if (pending_sep != "") {
        print pending_sep
    }

    if (!added_collection) {
        print "WARNING: Could not find active '" old ",' inside COLLECTIONS block" > "/dev/stderr"
    }

    if (!done_section) {
        print "ERROR: Could not find complete " old " section ending with ::" > "/dev/stderr"
        exit 2
    }
}
AWK

awk \
    -v old="$old" \
    -v new="$new" \
    -f "$awk_script" \
    "$file" > "$tmp"

mv "$tmp" "$file"

echo "Updated $file"
echo "Backup saved as ${file}.bak"