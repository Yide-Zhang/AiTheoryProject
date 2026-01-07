
import pooltool as pt
import inspect

print("Checking pooltool methods for saving...")
ms = pt.MultiSystem()
print(f"Has save method: {hasattr(ms, 'save')}")

s = pt.System()
print(f"System has save method: {hasattr(s, 'save')}")
