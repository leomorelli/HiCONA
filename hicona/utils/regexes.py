"""All regexes used in the package."""

# Genomic region regexes
POS_LIKE_STR = r"(chr[0-9]{1,2}|X|Y|MT):(\d+)*-(\d+)*"  # es. chr1:100-200
BED_LIKE_STR = r"(chr[0-9]{1,2}|X|Y|MT)[ |\t](\d+)*[ |\t](\d+)*"  # es. chr1 100 200
CHR_LIKE_STR = r"(chr[0-9]{1,2}|X|Y|MT)"  # es. chr1
