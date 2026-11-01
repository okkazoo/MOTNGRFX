=== CHECK YOUR REPLY BEFORE YOU ANSWER ===
- JSON only, every bracket closed, no comments, no trailing commas.
- Every reference to a buffer starts with `$`; every op that needs a mask gets a gray buffer, every op that needs a colour picture an RGB one; the last op writes `image`; no op reads a buffer a later op makes.
- Every font is a file name from the FONTS list exactly, at least three distinct faces, at least one expressive (pixel / ornamental / decon / blackletter / heavy-expressive); axes use that font's tags and ranges.
- Every `hit` is a time that appears in the MUSIC block; the first hit is at least its `form_s` after 0; `hold_s` >= 1.2; the last word's hold ends before the cut length minus 0.2 s; transitions: no name twice.
- Shots: `until` of the last shot = the piece length; cuts on downbeats; shot prompts differ and each states one camera move; `first` names a still you wrote or an earlier shot. Structure: one scene and one camera move = one long shot; a story with beats or places = several shots (a place of more than one shot continues with `first: "last:<id>"`), never one long shot for a piece that visits places.
- The type is not static: camera, yaw, an animated axis. The hits and holds are not all the same length. The chain is longer than a default and has at least two timelines tied to the music or the sections.
- No example value of this guide is copied; the numbers belong to THIS brief.
