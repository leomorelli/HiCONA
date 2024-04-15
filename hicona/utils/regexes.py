"""All regexes used in the package."""

# Genomic region regexes

# es. chr1:100-200
POS_LIKE_STR = r"(chr[0-9]{1,2}|chrX|chrY|chrMT):(\d+)*-(\d+)*"

# es. chr1 100 200
BED_LIKE_STR = r"(chr[0-9]{1,2}|chrX|chrY|chrMT)[ |\t](\d+)*[ |\t](\d+)*"

# es. chr1
CHR_LIKE_STR = r"(chr[0-9]{1,2}|chrX|chrY|chrMT)"
