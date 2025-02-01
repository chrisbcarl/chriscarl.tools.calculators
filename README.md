# chriscarl.tools.calculators
Tools that are calculators.

These types of projects are typically of low "software" quality but of high "readability" quality--not DRY, reimplements code in other libraries, uses prints, globals, long files, no dependencies other than the language, "intern-style", etc.


# Usage
```bash
python calculators/numeric.py  "1+1" "16 ^ 27 // (3 ** 2)" -c -v
python calculators/discrete.py "1+1" "16 ^ 27 // (3 ** 2)" -c -v
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
