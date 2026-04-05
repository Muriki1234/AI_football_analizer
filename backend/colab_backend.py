# This file re-exports from the canonical colab_backend.py at the project root.
# Run either file — they are the same app.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from colab_backend import app  # noqa: F401

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False, use_reloader=False)
