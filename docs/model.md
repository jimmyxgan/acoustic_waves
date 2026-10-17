# Physical and numerical model

## Equations

The solver uses first-order linear acoustics for stationary homogeneous air:

- `dv/dt = -(1/rho) grad(p)`
- `dp/dt = -rho*c^2 div(v)`

Pressure is stored at cell centers; velocity components are face-centered. The timestep is selected to satisfy the nominal 3D acoustic CFL limit.

## Boundary treatment

The epoxy sidewalls are treated as rigid except for the localized prescribed wall-normal velocity source. Exterior computational boundaries use a numerical sponge, not a perfectly matched layer. The y direction represents a finite computational segment of a much larger annular channel and terminates in absorbing boundaries rather than artificial rigid walls.

## Source

An exponentially damped sinusoid drives a small patch of the epoxy wall. Its amplitude is a hypothetical wall velocity; the simulation does not determine wall motion from electrical apparent charge or structural mechanics.

## Interpretation

The 3D solver captures spreading, reflections, and interference within its simplified geometry. Saved animation frames are 2D slices only. Quantitative microphone pressure depends strongly on assumed source amplitude, geometric dimensions, numerical resolution, and boundary treatment.

## Before presenting quantitative claims

1. Verify grid convergence at the frequency of interest.
2. Test sensitivity to sponge thickness, strength, and external-domain extent.
3. Test source patch diameter and circumferential extent.
4. Incorporate or separately bound atmospheric ultrasonic attenuation.
5. Avoid claims of calibrated detection sensitivity without measured PD-to-acoustic coupling and microphone frequency response.

## Four rigid duct boundaries (updated)

Inside `0 <= z <= H`, the rectangular air region is `0 <= x <= W` and
`-D/2 <= y <= D/2`. All four longitudinal walls are rigid, except for the
localized prescribed normal-velocity source on the x=0 epoxy wall. The y walls
represent adjacent rigid duct sticks. Above and below the duct, the model
contains external air and a numerical damping sponge at the far domain edges.
The sponge is not a physical wall loss; it approximates outward radiation.
The microphone is in undamped external air below the lower opening.

The computational box in x/y is finite and should be large enough that the
sponge does not interfere with radiation near the opening. No physical
atmospheric or viscothermal absorption is modeled. Pressure at the microphone
is normalized to an assumed wall velocity and does not establish PD pC sensitivity.
