# SoundFont banks used by `mograph.sampled`

Banks are downloaded once by `mograph.banks.ensure_banks()` into `~/.cache/mograph/banks` (override with
`MOGRAPH_BANKS`). They are never committed to this repo. Each file is checked against a pinned SHA-256 after download.
Bank licences are separate from this repo's Apache-2.0 licence, and this repo does not relicense them or clear the
rights to any sample in them. Read the original notices before redistributing a bank or a render commercially.

The bank list, versions and hashes follow op55 (github.com/ynginkim/op55, MIT, `skills/op55/scripts/lib/banks.mjs`).
The hashes were re-verified here against the files actually downloaded (2026-10-24).

| Name | Bank | Licence | Credit required | Size | SHA-256 (of the .sf2) |
|---|---|---|---|---|---|
| `gm` | GeneralUser GS 2.0.3, S. Christian Collins | GeneralUser GS License v2.0 | no | 32,319,396 B | `9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe` |
| `upright` | Upright Piano KW 2022-02-21, FreePats project | CC0 1.0 | no | 57,377,848 B | `d9f5157720963671906727ca2e12b3293fd822c831bb0de477dd1c5f3ad37108` |
| `grand` | Salamander Grand Piano V3 2020-06-02, Alexander Holm | CC BY 3.0 | **yes** | ~1.27 GB | `712d0e681efbe5203a8014e9b3e84168f1908c82f2f6fb13bd2c77d6d72c70b7` (from op55; never downloaded or checked here) |

## gm: GeneralUser GS 2.0.3

- Summary: free for your own music creation, private or commercial; free to use in software projects and to modify.
  No credit required.
- Source used: the author's GitHub repo at a pinned commit,
  `https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/97049183643d5fc5a9322a69c5b09efb667c6c3a/GeneralUser-GS.sf2`.
  Fallback: op55's release copy (gzip), `https://github.com/ynginkim/op55/releases/download/banks-v1/GeneralUser-GS-2.0.3.sf2.gz`.
  The author asks sites not to link directly to the download files on his own website (schristiancollins.com), so
  neither source is his site. The SHA-256 above matched on the GitHub file.
- Licence text: https://github.com/mrbumpy409/GeneralUser-GS/blob/97049183643d5fc5a9322a69c5b09efb667c6c3a/documentation/LICENSE.txt
- **Sample-origin caveat (the author's own words, summarised):** many samples are original, but some were taken
  from other freely available SoundFont banks on the internet. Because the bank began as a personal project with no
  plan to publish, he cannot be 100 % sure where every sample came from, though none came from commercially
  published SoundFonts or sample CDs. He has never received a complaint, but says the uncertainty may matter if you
  ship it inside a commercial software product. Rendered music is the use he explicitly allows; bundling the bank
  inside a product is where the caveat bites.

## upright: Upright Piano KW (FreePats)

- Summary: CC0 1.0 public-domain dedication. No attribution needed.
- Recorded by Gonzalo and Roberto in January 2017, edited by Roberto. Page and licence statement:
  https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html
- Source used: op55's release copy (gzip), `https://github.com/ynginkim/op55/releases/download/banks-v1/UprightPianoKW-20220221.sf2.gz`.
  The uncompressed SHA-256 matched. Fallback (official, not exercised here):
  `https://freepats.zenvoid.org/Piano/UprightPianoKW/UprightPianoKW-SF2-20220221.7z`, which needs a `tar` that reads 7z.
- Legal code: https://creativecommons.org/publicdomain/zero/1.0/legalcode

## grand: Salamander Grand Piano V3 (optional, not downloaded)

- Registered in `mograph.banks.BANKS` with `auto=False`; `ensure_banks` skips it unless `allow_large=True`.
- CC BY 3.0, credit required. Put this line in the credits of any render that uses it:
  **Salamander Grand Piano by Alexander Holm (CC BY 3.0)**
- Source: `https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/SalamanderGrandPiano-SF2-V3%2B20200602.tar.xz`
  (about 300 MB download, 1.2 GB on disk).
- Licence: https://creativecommons.org/licenses/by/3.0/

## Synth engine

SoundFonts are rendered by `symusic` (MIT, Yikai Liao; includes the Prestosynth renderer), pinned to 0.5.9 in
`requirements-music.txt`. 0.6.0 installs on Windows but its `Synthesizer.render()` fails to return a buffer.
