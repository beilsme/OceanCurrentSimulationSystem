# Ocean Current Simulation System

A multi-language ocean current simulation framework combining C++, C# and Python. The C++ core handles heavy calculations, while the C# layer orchestrates the overall workflow and exposes APIs. Python scripts are used for data acquisition, processing and visualization.

## Repository structure

- **Source/** – all source code
    - `CppCore/` – core C++ implementation and bindings
    - `CSharpClient/` – C# projects including UI and domain logic
    - `PythonEngine/` – Python services for data processing and simulation helpers
    - `Scripts/` – helper build and setup scripts
- **Documentation/** – project documents and specifications
- **Configuration/** – runtime configuration files

## Quick start

1. Ensure you have **Python 3.8+** and **.NET 6+** installed.
2. On first launch run the setup script which installs the Python/C# bridge:

```bash
python setup_python_csharp_interface.py
```

3. Build and run the complete system using the provided startup script:

```bash
# Linux/Mac
./startup.sh

# Windows
startup.bat
```

This script checks dependencies, builds the C++ core, restores the C# projects and launches the Python engine alongside the main application.

## Additional scripts

The `Source/Scripts` folder contains utilities such as `install.sh` to set up the Python engine manually and `build_cpp.sh` to compile the C++ components.

## License

This project is released under the [MIT License](LICENSE).