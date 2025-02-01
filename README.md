# chriscarl.tools.calculators
Tools that are calculators.

These types of projects are typically of low "software" quality but of high "readability" quality--not DRY, reimplements code in other libraries, uses prints, globals, long files, no dependencies other than the language, "intern-style", etc.


# Usage
- calculators/numeric
```bash
# run multiple calculations and "show your work" and sanity check
python calculators/numeric.py  "1+1" "16 ^ 27 // (3 ** 2)" --verbose --compare

# ommit args for plain output
python calculators/discrete.py "1+1" "16 ^ 27 // (3 ** 2)"
```
- calculators/discrete
```bash
# run multiple calculations, mix and match operator symbols
python calculators/discrete.py "p" "p & q" "( q )" "p | q" "!p" "p -> q" "not q" "p iff q" "p|p" "p -> ~q" "q*~q" "p & q | r"

# output one calculation as only one csv
python calculators/discrete.py "a * b + c * d" --formats csv --latex --output-filepath "/tmp/out.csv"

# output one calculation as working latex html
python calculators/discrete.py "p and q or r" "(p & q) <-> (r imp -s)" --formats html --latex --expand --output-filepath "/tmp/out.html"

# output one calculation as html, markdown, json, csv, use latex rather than unicode, expand the json and the html, and save all output (messy as it is)
python calculators/discrete.py "p and q or r" "(p & q) <-> (r imp -s)" --formats html md json csv --latex --expand --output-filepath "/tmp/out.txt"

# really nasty one
python calculators/discrete.py "(p -> q) | (r and -s) iff (t and ~u) implies v"
```


# Setup
1. install python of any version. (terminal examples assume one version)
    - web:
        - [3.12.8](https://www.python.org/downloads/release/python-3128/)
        - [3.8.6](https://www.python.org/downloads/release/python-386/)
        - [2.7.18](https://www.python.org/downloads/release/python-2718/)
    - windows:
        ```powershell
        # https://chocolatey.org/install
        # open in administrator
        Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        # open new powershell in admin to refresh env vars
        choco install python --version 3.12.8 --yes
        ```
    - debian:
        ```bash
        sudo apt update -y
        sudo apt install python3.12 -y
        alias python='python3.12'
        # alias will disappear on new terminal, add to .bashrc or other means
        ```
    - fedora/centos
        ```bash
        sudo dnf clean all -y
        sudo dnf update -y
        dnf install python3 -7
        alias python='python3.12'
        # alias will disappear on new terminal, add to .bashrc or other means
        ```


# Authors
- Chris Carl <chrisbcarl@outlook.com>
