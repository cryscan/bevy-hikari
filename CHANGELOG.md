# Changelog

## [0.3.15] - 2022-12-24
### Changed
- Make enabling/disabling emissive spatial reuse a separate config item (default to false).

### Fixed
- Fix TAA leaving trials on the background.

## [0.3.14] - 2022-12-11
### Changed
- Make SMAA Tu4x ratio 2.0 the default setting.

## [0.3.10] - 2022-12-8
### Changed
- SMAA now re-projects and clamps previous frame pixels in a much more cleaver way,
  which greatly reduces ghosting artefacts.

### Fixed
- Fix nearest velocity implementation.

## [0.3.9] - 2022-12-7
### Changed
- SMAA now rejects samples re-projected from the previous frame also by matching sub-pixel velocity with jittering.
  This allows less aliasing for upscaled renderings in motion.
- Use nearest velocity in both SMAA and TAA.

## [0.3.8] - 2022-12-6
### Added
- Add type `HikariUniversalSettings` for disabling building of acceleration structures. ([#35](https://github.com/cryscan/bevy-hikari/issues/35))

### Fixed
- Fix more TAA ghosting for multiple upscaling ratios.

## [0.3.7] - 2022-12-5
This is mainly a bug-fix and performance upgrade.

### Changed
- Optimize data structures and bandwidth usage.

### Fixed
- Fix SMAA previous color re-projection to get less aliasing when moving.
- Fix TAA when the upscaling ratio is 1.0.

## [0.3.5] - 2022-12-4
### Changed
- Do TAA color clipping on instance miss. ([b736703](https://github.com/cryscan/bevy-hikari/tree/b73670378dab3103e4e5603e5d7e60af76188906))

### Fixed
- Suppress bloom fireflies. ([252ee68](https://github.com/cryscan/bevy-hikari/tree/252ee68964e97574c6c43f5ca7b215697e515546))

## [0.3.4] - 2022-12-2
### Changed
- Apply firefly filtering to emissive lighting. ([e7c8c52](https://github.com/cryscan/bevy-hikari/tree/e7c8c52526404724d8a2d6d55e22bc62f0e9ae02))
- Invalidate previous spatial reservoir if the corresponding temporal reservoir is invalidated to speedup light updating.

## [0.3.3] - 2022-12-1
### Changed
- Improve SMAA Tu4x quality when moving by sampling previous full scale rendering. ([065ca1c](https://github.com/cryscan/bevy-hikari/tree/065ca1ce5d10c3dca69f415e7a9c46072160a68a))
- Unroll loops in denoiser. ([4ecfbf5](https://github.com/cryscan/bevy-hikari/tree/4ecfbf54425142ac934df7dcb9759209f95c4e6e))

### Fixed
- Spatial reusing kernel jitter each frame to avoid non-convergence. ([aca0001](https://github.com/cryscan/bevy-hikari/tree/aca00016e9b99d3582b73ea51b93cb54fdf50779))

## [0.3.2] - 2022-11-29
### Changed
- **Breaking:** Add `hikari` render graph back.
  Users need to set `camera_render_graph` in their `Camera3dBundle` to `bevy_hikari::graph::NAME` in order to activate path tracing.
- Bump `bevy` from v0.9.0 to v0.9.1. ([863c73f](https://github.com/cryscan/bevy-hikari/tree/863c73fe5f649dc2a670eb6cae6817e02c6a1973))
- Change how TAA deals with dis-occlusion from color clipping to exponential blending.
  This introduces some ghosting but yields better upscaled results.
- Render prepass textures (G-Buffers) in upscaled resolution. ([bb37f6a](https://github.com/cryscan/bevy-hikari/tree/bb37f6a7d085edd475d2142bfcdc5a3176ee3e10))

### Added
- Construct and sample light BVH when rendering emissive lights. ([#86](https://github.com/cryscan/bevy-hikari/pull/86))
- Firefly filtering for indirect denoiser. ([8e87987](https://github.com/cryscan/bevy-hikari/tree/8e8798768f082233d8b8c39fcabff4a47fccb38e))

### Removed
- Disable denoiser's temporal accumulation. ([97c4081](https://github.com/cryscan/bevy-hikari/tree/97c4081df6dee24d6e11df2ea0059a4126795d62))

[0.3.15]: https://github.com/cryscan/bevy-hikari/commits/v0.3.15
[0.3.14]: https://github.com/cryscan/bevy-hikari/commits/v0.3.14
[0.3.10]: https://github.com/cryscan/bevy-hikari/commits/v0.3.10
[0.3.9]: https://github.com/cryscan/bevy-hikari/commits/v0.3.9
[0.3.8]: https://github.com/cryscan/bevy-hikari/commits/v0.3.8
[0.3.7]: https://github.com/cryscan/bevy-hikari/commits/v0.3.7
[0.3.5]: https://github.com/cryscan/bevy-hikari/commits/v0.3.5
[0.3.4]: https://github.com/cryscan/bevy-hikari/commits/v0.3.4
[0.3.3]: https://github.com/cryscan/bevy-hikari/commits/v0.3.3
[0.3.2]: https://github.com/cryscan/bevy-hikari/commits/v0.3.2