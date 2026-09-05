"""
M87* EHT Stokes-I reconstruction from calibrated UVFITS.

Educational imaging reconstruction. This is NOT the official EHT imaging
pipeline and is not intended to reproduce the published EHT image exactly.

Input:
    SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits

Dependencies:
    numpy
    scipy
    matplotlib
    astropy

Run:
    python m87_reconstruct.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import minimize


# ============================================================
# SETTINGS
# ============================================================

UVFITS_PATH = Path(
    "data/EHTC_FirstM87Results_Apr2019/uvfits/"
    "SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits"
)

FOV_UAS = 128.0
NPIX = 256

MAXITER = 300

# Regularization.
ENTROPY_WEIGHT = 0.002
SMOOTHNESS_WEIGHT = 0.01

# Limit number of visibilities if memory/time is an issue.
# None = use all.
MAX_VIS = 3000

EPS = 1e-12


# ============================================================
# CONSTANTS
# ============================================================

C = 299792458.0

RAD_PER_UAS = np.pi / (180.0 * 3600.0 * 1e6)


# ============================================================
# LOAD UVFITS
# ============================================================

if not UVFITS_PATH.exists():
    raise FileNotFoundError(
        f"\nCould not find:\n{UVFITS_PATH}\n\n"
        "Change UVFITS_PATH at the top of the script."
    )


print("=" * 70)
print("M87* EHT RECONSTRUCTION")
print("=" * 70)
print("File:", UVFITS_PATH)


with fits.open(UVFITS_PATH, memmap=False) as hdul:

    hdu = hdul[0]
    header = hdu.header
    data = hdu.data

    print("HDU type:", type(data).__name__)

    # --------------------------------------------------------
    # Random-groups UVFITS
    # --------------------------------------------------------

    if not hasattr(data, "parnames"):
        raise RuntimeError(
            "This does not appear to be a UVFITS random-groups file."
        )

    print("Group parameters:", data.parnames)

    # UV coordinates are stored as group parameters.
    parnames = list(data.parnames)

    def find_parameter(name):
        for p in parnames:
            if p.upper() == name.upper():
                return p
        return None

    u_name = find_parameter("UU---SIN")
    v_name = find_parameter("VV---SIN")

    if u_name is None:
        u_name = find_parameter("UU")

    if v_name is None:
        v_name = find_parameter("VV")

    if u_name is None or v_name is None:
        raise RuntimeError(
            "Could not find UU/VV coordinates.\n"
            f"Available parameters: {parnames}"
        )

    u_sec = np.asarray(
        data.par(u_name),
        dtype=np.float64
    )

    v_sec = np.asarray(
        data.par(v_name),
        dtype=np.float64
    )

    # The actual visibility data.
    raw = np.asarray(data.data)

    print("Raw visibility array shape:", raw.shape)
    print("Raw visibility dtype:", raw.dtype)

    if raw.ndim < 2:
        raise RuntimeError(
            f"Unexpected UVFITS DATA shape: {raw.shape}"
        )

    # The first dimension corresponds to groups/visibilities.
    if raw.shape[0] != len(u_sec):
        raise RuntimeError(
            "Number of UV coordinates does not match "
            "number of visibility groups.\n"
            f"UV points: {len(u_sec)}\n"
            f"DATA groups: {raw.shape[0]}"
        )

    # --------------------------------------------------------
    # Determine the visibility axes.
    #
    # The final axes normally contain:
    #
    #   visibility component: 3
    #       real, imaginary, weight
    #
    #   Stokes: 1 or 4
    #
    #   frequency: possibly 1 or more
    #
    # We use the first frequency and Stokes-I.
    # --------------------------------------------------------

    image_shape = raw.shape[1:]

    print("Per-group DATA shape:", image_shape)

    # Find an axis of length 3 for real/imag/weight.
    component_axes = [
        i for i, n in enumerate(image_shape)
        if n == 3
    ]

    if not component_axes:
        raise RuntimeError(
            "Could not find the real/imag/weight axis in DATA."
        )

    # In standard UVFITS this is the last axis in Python order,
    # but we detect it rather than assuming it.
    comp_axis = component_axes[-1]

    print("Component axis:", comp_axis)

    # Move component axis to the end.
    arr = np.moveaxis(
        raw,
        comp_axis + 1,
        -1
    )

    # After this:
    #
    # arr.shape = (Nvis, ..., 3)
    #
    # Collapse all remaining axes except component.
    arr = arr.reshape(
        arr.shape[0],
        -1,
        3
    )

    # Select the first available polarization/frequency cell.
    # For the supplied Stokes-I release this corresponds to the
    # Stokes-I visibility data.
    stokes_I = arr[:, 0, :]

    real = np.asarray(
        stokes_I[:, 0],
        dtype=np.float64
    )

    imag = np.asarray(
        stokes_I[:, 1],
        dtype=np.float64
    )

    weight = np.asarray(
        stokes_I[:, 2],
        dtype=np.float64
    )


# ============================================================
# FREQUENCY
# ============================================================

# Prefer explicit observing-frequency keywords.
frequency = None

for key in (
    "CRVAL4",
    "RESTFREQ",
    "FREQ",
    "OBSFREQ",
):
    if key in header:
        try:
            value = float(header[key])

            # Frequency must be physically plausible.
            if 1e9 < value < 1e12:
                frequency = value
                break

        except Exception:
            pass


if frequency is None:
    # EHT 2017 1.3 mm observations.
    frequency = C / 1.3e-3

    print(
        "WARNING: frequency keyword not found; "
        "using 230 GHz."
    )


print("Frequency:", frequency / 1e9, "GHz")


# ============================================================
# CLEAN DATA
# ============================================================

good = (
    np.isfinite(u_sec)
    & np.isfinite(v_sec)
    & np.isfinite(real)
    & np.isfinite(imag)
    & np.isfinite(weight)
    & (weight > 0)
)

u_sec = u_sec[good]
v_sec = v_sec[good]
real = real[good]
imag = imag[good]
weight = weight[good]

vis = real + 1j * imag


print("Valid visibilities:", len(vis))


if len(vis) == 0:
    raise RuntimeError("No valid visibilities found.")


# ============================================================
# CONVERT UV TO WAVELENGTHS
# ============================================================

u = u_sec * frequency
v = v_sec * frequency

baseline = np.hypot(u, v)


print(
    "Baseline range:",
    baseline.min() / 1e9,
    "to",
    baseline.max() / 1e9,
    "Glambda"
)


# ============================================================
# REDUCE DATASET
# ============================================================

if MAX_VIS is not None and len(vis) > MAX_VIS:

    rng = np.random.default_rng(12345)

    take = rng.choice(
        len(vis),
        MAX_VIS,
        replace=False
    )

    u = u[take]
    v = v[take]
    vis = vis[take]
    weight = weight[take]

    baseline = baseline[take]

    print("Using random subset:", len(vis))


# ============================================================
# UVFITS CONVENTION
# ============================================================
#
# UVFITS baseline coordinates use the opposite sign convention
# from the common Fourier imaging convention. We therefore flip
# u/v here.
#
# This is consistent with the convention used by pyuvdata when
# reading UVFITS.
# ============================================================

u = -u
v = -v


# ============================================================
# HERMITIAN UV COVERAGE
# ============================================================
#
# Real-valued sky brightness implies:
#
# V(-u,-v) = conjugate(V(u,v))
#
# Explicitly add the conjugate measurements.
# ============================================================

u = np.concatenate([u, -u])
v = np.concatenate([v, -v])

vis = np.concatenate([vis, np.conj(vis)])
weight = np.concatenate([weight, weight])

baseline = np.hypot(u, v)


print("After Hermitian completion:", len(vis))


# ============================================================
# IMAGE GRID
# ============================================================

fov_rad = FOV_UAS * RAD_PER_UAS
pixel_rad = fov_rad / NPIX

axis_rad = (
    np.arange(NPIX) -
    (NPIX - 1) / 2.0
) * pixel_rad

X, Y = np.meshgrid(
    axis_rad,
    axis_rad,
    indexing="xy"
)

Xf = X.ravel()
Yf = Y.ravel()

n_pix = NPIX * NPIX

pixel_area = pixel_rad ** 2


print("Image:", NPIX, "x", NPIX)
print("FOV:", FOV_UAS, "uas")
print("Pixel:", pixel_rad / RAD_PER_UAS, "uas")


# ============================================================
# NORMALIZE WEIGHTS
# ============================================================

weight = np.maximum(weight, 0)

weight /= np.median(weight)

weight_sum = np.sum(weight)


# ============================================================
# FLUX INITIALIZATION
# ============================================================
# Estimate total flux from the shortest baselines rather than
# assuming the maximum visibility equals the total flux.
# ============================================================

short_limit = np.percentile(
    baseline,
    5
)

short = baseline <= short_limit

if np.any(short):
    total_flux0 = np.median(
        np.abs(vis[short])
    )
else:
    total_flux0 = np.max(
        np.abs(vis)
    )


if not np.isfinite(total_flux0) or total_flux0 <= 0:
    total_flux0 = 1.0


print("Initial flux:", total_flux0, "Jy")


# ============================================================
# FOURIER TRANSFORM
# ============================================================
#
# Instead of storing an enormous A matrix, calculate the
# Fourier transform in chunks.
# ============================================================

CHUNK = 256


def forward(image):
    """
    Calculate model complex visibilities.

    image is surface brightness in Jy/sr.
    """

    img = image.ravel()

    result = np.empty(
        len(u),
        dtype=np.complex128
    )

    for start in range(
        0,
        len(u),
        CHUNK
    ):

        stop = min(
            start + CHUNK,
            len(u)
        )

        phase = (
            -2j * np.pi *
            (
                u[start:stop, None] * Xf[None, :]
                +
                v[start:stop, None] * Yf[None, :]
            )
        )

        result[start:stop] = (
            np.exp(phase) @ img
        ) * pixel_area

    return result


def adjoint(residual):
    """
    Adjoint Fourier operator.
    """

    result = np.zeros(
        n_pix,
        dtype=np.complex128
    )

    for start in range(
        0,
        len(u),
        CHUNK
    ):

        stop = min(
            start + CHUNK,
            len(u)
        )

        phase = (
            2j * np.pi *
            (
                u[start:stop, None] * Xf[None, :]
                +
                v[start:stop, None] * Yf[None, :]
            )
        )

        result += (
            np.exp(phase).T @ residual[start:stop]
        )

    return result * pixel_area


# ============================================================
# INITIAL IMAGE
# ============================================================

sigma0 = 15.0 * RAD_PER_UAS

gaussian = np.exp(
    -(X**2 + Y**2) /
    (2.0 * sigma0**2)
)

gaussian /= (
    gaussian.sum() * pixel_area
)

image0 = (
    total_flux0 *
    gaussian
)


# ============================================================
# IMAGE PARAMETERIZATION
# ============================================================
#
# q -> positive image.
#
# log_flux -> total flux.
# ============================================================

def unpack(z):

    q = z[:-1]
    log_flux = z[-1]

    q = q - np.max(q)

    p = np.exp(
        np.clip(q, -50, 50)
    )

    p_sum = np.sum(p)

    if p_sum <= 0:
        p[:] = 1.0
        p_sum = p.size

    p /= p_sum

    flux = np.exp(
        np.clip(log_flux, -10, 10)
    )

    # p is Jy/pixel.
    image = (
        flux *
        p /
        pixel_area
    )

    return image, flux


q0 = np.log(
    np.maximum(
        image0.ravel() * pixel_area,
        EPS
    )
)

z0 = np.concatenate(
    [
        q0,
        [np.log(total_flux0)]
    ]
)


# ============================================================
# REGULARIZATION
# ============================================================

def entropy(image):

    p = (
        image *
        pixel_area
    )

    p /= (
        np.sum(p) +
        EPS
    )

    p = np.clip(
        p,
        EPS,
        None
    )

    # Negative entropy is minimized.
    return np.sum(
        p * np.log(p)
    )


def smoothness(image):

    image2d = np.asarray(image).reshape(NPIX, NPIX)

    dx = (
        image2d[:, 1:] -
        image2d[:, :-1]
    )

    dy = (
        image2d[1:, :] -
        image2d[:-1, :]
    )

    scale = np.mean(image2d) + EPS

    return (
        np.mean((dx / scale) ** 2)
        +
        np.mean((dy / scale) ** 2)
    )



# ============================================================
# OBJECTIVE + ANALYTIC GRADIENT
# ============================================================

eval_counter = 0


def objective_and_gradient(z):

    global eval_counter

    eval_counter += 1

    image, flux = unpack(z)

    model = forward(image)

    residual = (
        model -
        vis
    )

    # Weighted complex chi-square.
    chi2 = np.sum(
        weight *
        (
            residual.real ** 2 +
            residual.imag ** 2
        )
    ) / (
        weight_sum
    )

    ent = entropy(image)

    smooth = smoothness(image)

    value = (
        chi2
        + ENTROPY_WEIGHT * ent
        + SMOOTHNESS_WEIGHT * smooth
    )

    # --------------------------------------------------------
    # Gradient of chi-square.
    # --------------------------------------------------------

    weighted_residual = (
        weight *
        residual
    )

    grad_image = (
        2.0 *
        np.real(
            adjoint(
                weighted_residual
            )
        )
        / weight_sum
    )

    # --------------------------------------------------------
    # Convert image gradient into q gradient.
    #
    # I = flux * softmax(q) / pixel_area
    # --------------------------------------------------------

    p = np.exp(
        np.clip(
            z[:-1] - np.max(z[:-1]),
            -50,
            50
        )
    )

    p /= p.sum()

    grad_pixel = (
        grad_image *
        flux /
        pixel_area
    )

    mean_grad = np.sum(
        p *
        grad_pixel
    )

    grad_q = (
        p *
        (
            grad_pixel -
            mean_grad
        )
    )

    # Flux derivative.
    #
    # dI/d(log flux) = I
    #
    grad_log_flux = np.sum(
        grad_image *
        image
    )

    gradient = np.concatenate(
        [
            grad_q,
            [grad_log_flux]
        ]
    )

    if eval_counter % 10 == 0:

        print(
            f"eval {eval_counter:4d} | "
            f"objective={value:.6g} | "
            f"chi2={chi2:.6g} | "
            f"flux={flux:.4f} Jy"
        )

    return (
        float(value),
        gradient
    )


# ============================================================
# OPTIMIZATION
# ============================================================

print()
print("=" * 70)
print("STARTING OPTIMIZATION")
print("=" * 70)

result = minimize(
    objective_and_gradient,
    z0,
    method="L-BFGS-B",
    jac=True,
    options={
        "maxiter": MAXITER,
        "ftol": 1e-9,
        "gtol": 1e-6,
        "maxls": 20,
        "maxcor": 10,
    }
)


print()
print("=" * 70)
print("OPTIMIZATION FINISHED")
print("=" * 70)

print("Success:", result.success)
print("Message:", result.message)
print("Iterations:", result.nit)
print("Function evaluations:", result.nfev)


# ============================================================
# FINAL IMAGE
# ============================================================

image, recovered_flux = unpack(
    result.x
)

image = image.reshape(
    NPIX,
    NPIX
)

image_display = (
    image /
    np.max(image)
)


# ============================================================
# FINAL MODEL
# ============================================================

model = forward(
    image
)

residual = (
    model -
    vis
)

chi2 = np.sum(
    weight *
    (
        residual.real ** 2 +
        residual.imag ** 2
    )
) / weight_sum


print()
print("Final chi-square:", chi2)
print("Recovered flux:", recovered_flux, "Jy")
print(
    "Peak brightness:",
    np.max(image),
    "Jy/sr"
)


# ============================================================
# DIRTY IMAGE
# ============================================================

print("Making dirty image...")


dirty = np.zeros(
    n_pix,
    dtype=np.complex128
)

beam = np.zeros(
    n_pix,
    dtype=np.complex128
)

for start in range(
    0,
    len(u),
    CHUNK
):

    stop = min(
        start + CHUNK,
        len(u)
    )

    phase = (
        2j * np.pi *
        (
            u[start:stop, None] * Xf[None, :]
            +
            v[start:stop, None] * Yf[None, :]
        )
    )

    ww = weight[start:stop]

    dirty += (
        np.exp(phase).T @
        (
            ww *
            vis[start:stop]
        )
    )

    beam += (
        np.exp(phase).T @
        ww
    )


dirty = (
    dirty.real /
    (np.sum(weight) + EPS)
)

beam = (
    beam.real /
    (np.sum(weight) + EPS)
)


dirty = dirty.reshape(
    NPIX,
    NPIX
)

beam = beam.reshape(
    NPIX,
    NPIX
)


# ============================================================
# PLOTS
# ============================================================

extent = [
    axis_rad[0] / RAD_PER_UAS,
    axis_rad[-1] / RAD_PER_UAS,
    axis_rad[0] / RAD_PER_UAS,
    axis_rad[-1] / RAD_PER_UAS,
]


# ------------------------------------------------------------
# Reconstruction
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    image_display,
    origin="lower",
    extent=extent,
    cmap="inferno"
)

plt.xlabel("Relative RA (μas)")
plt.ylabel("Relative Dec (μas)")
plt.title("M87* Regularized Reconstruction")

plt.colorbar(
    label="Normalized intensity"
)

plt.tight_layout()

plt.savefig(
    "m87_reconstructed.png",
    dpi=200
)


# ------------------------------------------------------------
# Dirty image
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 7)
)

dirty_display = (
    dirty /
    np.max(
        np.abs(dirty)
    )
)

plt.imshow(
    dirty_display,
    origin="lower",
    extent=extent,
    cmap="inferno"
)

plt.xlabel("Relative RA (μas)")
plt.ylabel("Relative Dec (μas)")
plt.title("M87* Dirty Image")

plt.colorbar(
    label="Normalized intensity"
)

plt.tight_layout()

plt.savefig(
    "m87_dirty.png",
    dpi=200
)


# ------------------------------------------------------------
# Dirty beam
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 7)
)

beam_display = (
    beam /
    np.max(
        np.abs(beam)
    )
)

plt.imshow(
    beam_display,
    origin="lower",
    extent=extent,
    cmap="gray"
)

plt.xlabel("Relative RA (μas)")
plt.ylabel("Relative Dec (μas)")
plt.title("Dirty Beam")

plt.colorbar(
    label="Normalized response"
)

plt.tight_layout()

plt.savefig(
    "m87_dirty_beam.png",
    dpi=200
)


# ------------------------------------------------------------
# Visibility amplitudes
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    baseline / 1e9,
    np.abs(vis),
    s=3,
    alpha=0.25,
    label="Data"
)

plt.scatter(
    baseline / 1e9,
    np.abs(model),
    s=3,
    alpha=0.25,
    label="Model"
)

plt.xlabel("Baseline length (Gλ)")
plt.ylabel("Visibility amplitude (Jy)")
plt.title("M87* Visibility Amplitudes")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "m87_visibility.png",
    dpi=200
)


# ------------------------------------------------------------
# Residuals
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    baseline / 1e9,
    np.abs(residual),
    s=3,
    alpha=0.3
)

plt.xlabel("Baseline length (Gλ)")
plt.ylabel("|Data - Model| (Jy)")
plt.title("Visibility Residuals")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "m87_residuals.png",
    dpi=200
)


# ============================================================
# SAVE DATA
# ============================================================

np.save(
    "m87_reconstructed.npy",
    image
)

np.savetxt(
    "m87_reconstructed.txt",
    image
)

np.save(
    "m87_dirty.npy",
    dirty
)

np.save(
    "m87_dirty_beam.npy",
    beam
)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print("Saved:")
print("  m87_reconstructed.npy")
print("  m87_reconstructed.txt")
print("  m87_reconstructed.png")
print("  m87_dirty.npy")
print("  m87_dirty.png")
print("  m87_dirty_beam.npy")
print("  m87_dirty_beam.png")
print("  m87_visibility.png")
print("  m87_residuals.png")

print()
print("Recovered total flux:", recovered_flux, "Jy")
print("Final chi-square:", chi2)

plt.show()
