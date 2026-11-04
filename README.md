# Bleach: Soul Resurrección Tools

A collection of open-source tools for modifying the PlayStation 3 game *Bleach: Soul Resurrección*, developed by the community. I should not take credit.

With the exception of Resurreccion Toolkit, these tools were recovered from archived forums that are no longer accessible. The files have only been tested with the American version of the game (BLUS30769).

## Credits

- **Xandreale** — Creator of [**BSIgnition**](https://archive.vg-resource.com/thread-21407-post-504639.html#pid504639), a tool developed in 2011 using FBSL (Freestyle BASIC Scripting Language).
- **neff10** — Creator of a tutorial explaining how to extract `.SPK` models and textures using **BSIgnition**.
- [**akderebur**](https://github.com/akderebur) — Creator of [**BSRViewer**](https://web.archive.org/web/20210125003805/https://forum.xentax.com/viewtopic.php?f=16&t=22570), a tool capable of reading 3D models, bones, and animations, playing animations back, and exporting models and animations to `.NUX` format for use with Noesis.
- [**SockNastre**](https://github.com/SockNastre) — Creator of [**Resurreccion Toolkit**](https://github.com/SockNastre/Resurreccion-Toolkit), who provided valuable information on modifying battle characters.

## Brief Guide

Inside the game's `USRDIR` directory, you will find an `SPK` folder containing files with the `.SPK` extension. These files require the tools provided in this repository to be read and modified.

Character files with the `figure` prefix contain more detailed character models and are used in the game's Collection menu.

Character files with the `btl` prefix, located in the `sndEnglish` and `sndJapanese` directories, contain the character assets used during gameplay. Both directories contain copies of the same character files, differing only in their audio language. Therefore, any modifications must be applied to the files in both directories.

**Example:** To modify Bankai Ichigo's in-game texture, you must edit both `spk/sndEnglish/btlIchigoImg.spk` and `spk/sndJapanese/btlIchigoImg.spk`.

Textures use the DDS BC1/DXT1 format with mipmaps (10 levels, which is the default setting in GIMP and Paint.NET). When modifying these textures, the final exported file must be exactly the same size as the corresponding original file.

Extract the desired `.SPK` file using `spk_extractor.py`, modify the target texture while preserving its original format, dimensions, and file size, and use `spk_inject.py` (the .dds textures must be in the same folder) to replace the original texture with the modified one.

### Usage

```bash
python spk_extractor.py btlIchigoImg.spk
python spk_inject.py btlIchigoImg.spk
```
