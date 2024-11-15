
[README (中文)](README.md)
[README (English by LLMs)](README_EN.md)

# Shortcuts

[toc]

## Windows Context Menu Support

#### Introduction

- Adds a right-click menu option for scripts that can be applied directly to files or folders.
- Currently, only the ability to add scripts is supported, not the ability to remove them.
  - Removal can be accomplished using the open-source software [ContextMenuManager](https://github.com/BluePointLilac/ContextMenuManager).

#### Running

1. Choose the script you need, for example, ImageResizer_256x144.
2. Edit the Bat_ImageResizer_256x144.bat file in the ContextMenu directory:
   1. Set the Python path.
   2. (Optional) Modify the option name.
3. Run Bat_ImageResizer_256x144.bat with administrator privileges.

- Before modification:

```bat
@REM Please set the Python path.
set "python_path="

@REM Please set the option name.
set "menu_name=ScriptName"
```

- After modification:

```bat
@REM Please set the Python path.
set "python_path=F:\0_DATA\2_CODE\Anaconda\envs\asdTools\python.exe"

@REM Please set the option name.
set "menu_name=ResizeImageTo256x144"
```

#### Supported Scripts

1. ImageResizer_256x144
   * Resize images to 256x144.
   * Known bug: Cannot handle paths with spaces.
2. SDStylesBeautify
   - Beautify the styles.csv of the Stable Diffusion Web UI (including sorting, adding category headers, removing empty lines).
