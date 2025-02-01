# chriscarl.tools.calculators
Tools that are calculators.

These types of projects are typically of low "software" quality but of high "readability" quality--not DRY, reimplements code from other libraries, uses prints, globals, long files, no dependencies other than the language, "intern-style", etc.


# Usage
- `calculators/numeric.py`
```bash
# run multiple calculations and "show your work" and sanity check
python calculators/numeric.py  "1+1" "16 ^ 27 // (3 ** 2)" --verbose --compare

# ommit args for plain output
python calculators/discrete.py "1+1" "16 ^ 27 // (3 ** 2)"
```
- `calculators/discrete.py`
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
        - [latest](https://www.python.org/downloads/)
        - [3.12.8](https://www.python.org/downloads/release/python-3128/)
        - [3.8.6](https://www.python.org/downloads/release/python-386/)
        - [2.7.18](https://www.python.org/downloads/release/python-2718/)
    - windows:
        ```powershell
        # open as administrator
        # https://chocolatey.org/install
        Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        ```
        ```powershell
        # open new powershell as administrator (using the same terminal as above will cause the following to fail because choco is not on the PATH)
        choco install python3 --yes
        ```
    - debian:
        ```bash
        sudo apt update -y
        sudo apt install python3 -y
        alias python='python3'
        # alias will disappear on new terminal, add to .bashrc or other means
        ```
    - fedora/centos
        ```bash
        sudo dnf clean all -y
        sudo dnf update -y
        dnf install python3 -7
        alias python='python3'
        # alias will disappear on new terminal, add to .bashrc or other means
        ```
2. download the repository
    - zip
        - [click to download .zip](./archive/refs/heads/main.zip)
        - extract zip
    - git
        ```bash
        git clone https://github.com/chrisbcarl/chriscarl.tools.calculators.git
        ```
3. run the code
    - navigate to where you extracted or cloned
    - open a terminal in that directory
    - `python calculators/discrete.py --help`
4. profit


# Authors
- Chris Carl <chrisbcarl@outlook.com>
