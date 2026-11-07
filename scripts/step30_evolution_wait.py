"""Copy to once_wait.py only if initial resources require the approved wait.

The same process runs the unique controller and its resource wait, using its
ledger. The controller deletes only the bound copy after a real first update.
This template is retained; it is never a second dispatcher or an automatic retry.
"""
from step30_evolution_control import main

if __name__ == '__main__':
    main()
