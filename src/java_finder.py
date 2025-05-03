# utils/java_finder.py
import os, platform, subprocess
from pathlib import Path

def path_exists(p: str) -> bool:
    """Cheap cross-platform 'does this directory exist?'."""
    try:
        return Path(p).expanduser().is_dir()
    except OSError:
        return False


def _run_silent(cmd: str) -> str | None:
    """Return stdout (stripped) or None if the command fails."""
    try:
        out = subprocess.check_output(cmd, shell=True, text=True,
                                      stderr=subprocess.DEVNULL)
        return out.strip() or None
    except subprocess.CalledProcessError:
        return None


def find_java11() -> str:
    """
    Locate a Java-11 home directory, mirroring the logic of the
    original async JS helper.

    Search order:
      1. $JAVA11_HOME env-var
      2. macOS: /usr/libexec/java_home -v 11
      3. well-known install paths on Linux / macOS / Windows
    Raises RuntimeError if nothing is found.
    """
    # 1) explicit env-var
    env_home = os.getenv("JAVA11_HOME")
    if env_home and path_exists(env_home):
        return env_home

    # 2) macOS's /usr/libexec helper
    if platform.system() == "Darwin":
        found = _run_silent("/usr/libexec/java_home -v 11")
        if found and path_exists(found):
            return found

    # 3) common locations
    guesses = [
        "/usr/lib/jvm/temurin-11-jdk-amd64",                # Ubuntu / Debian (Temurin PPA)
        "/usr/lib/jvm/java-11-openjdk-amd64",               # distro openjdk
        "/Library/Java/JavaVirtualMachines/adoptopenjdk-11.jdk/Contents/Home",  # macOS Intel
        "/opt/homebrew/opt/openjdk@11",                     # macOS Apple-Silicon (brew)
        r"C:\Program Files\Eclipse Adoptium\jdk-11",        # Windows Temurin
        r"C:\Program Files\Java\jdk-11",                    # Windows generic
    ]
    for g in guesses:
        if path_exists(g):
            return g

    raise RuntimeError("Java 11 not found – install it or set JAVA11_HOME.")


# quick manual test
if __name__ == "__main__":
    print("Java 11 HOME =", find_java11())
