<#
Author:      Chris Carl
Date:        2025-02-01
Email:       chrisbcarl@outlook.com

Description:
    test cases for discrete.py from an operating system perspective

Updated:
    2025-02-01 - chriscarl - discrete.ps1 added, functions 'like' a test case, but its really not since this code is intern-like
#>

# bad eggs
$baddies = @(
    'python calculators/discrete.py "and p"',
    'python calculators/discrete.py "and and p"',
    'python calculators/discrete.py "and not p"',
    'python calculators/discrete.py "not and p"',
    'python calculators/discrete.py "not not p"',
    'python calculators/discrete.py "not not p and"',
    'python calculators/discrete.py "p and"',
    'python calculators/discrete.py "p and not"',
    'python calculators/discrete.py "p and or"',
    'python calculators/discrete.py "p and or p"'
)
$baddies | ForEach-Object {
    Invoke-Expression $_
    if (0 -eq $LASTEXITCODE) {
        Write-Warning "$_ :FALSE POSITIVE!"
        exit $LASTEXITCODE
    }
}

# good eggs
$goodies = @(
    'python calculators/discrete.py "p" "( q )" "!p" "p & q" "p | q" "p -> q" "p iff q" "p -> ~q" "p & q | r"',
    'python calculators/discrete.py "a & not b" --formats csv --latex --output-filepath "ignoreme/out.csv"',
    'python calculators/discrete.py "p and q or r" "(p & q) <-> (r imp -s)" --formats html md json csv --latex --expand --output-filepath "ignoreme/out.txt"',
    'python calculators/discrete.py "(p & q) | (r and -s) iff t and not u"',
    'python calculators/discrete.py "(p & q) | (r and -s) iff (t and not u) implies v"'
)
$goodies | ForEach-Object {
    Invoke-Expression $_
    if (0 -ne $LASTEXITCODE) {
        Write-Warning "$_ :FALSE NEGATIVE!"
        exit $LASTEXITCODE
    }
}