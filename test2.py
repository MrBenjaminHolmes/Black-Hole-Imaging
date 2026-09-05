import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from scipy.optimize import minimize


# ============================================================
# SETTINGS
# ============================================================

FILE = Path(
    "data/EHTC_FirstM87Results_Apr2019/uvfits/"
    "SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits"
)

FOV_UAS = 128
NPIX = 256
MAXITER = 300

ENTROPY_WEIGHT = 0.002
SMOOTHNESS_WEIGHT = 0.01

CHUNK = 256
EPS = 1e-12

RAD_PER_UAS = np.pi / (180 * 3600 * 1e6)


# ============================================================
# LOAD DATA
# ============================================================

with fits.open(FILE, memmap=False) as hdul:
    hdu = hdul[0]
    data = hdu.data
    header = hdu.header

    u = np.asarray(data.par("UU---SIN"), float)
    v = np.asarray(data.par("VV---SIN"), float)

    raw = np.asarray(data.data)

    # For this specific Stokes-I file:
    stokes_I = raw[:, 0, 0, 0, 0, 0, :]

    real = stokes_I[:, 0]
    imag = stokes_I[:, 1]
    weight = stokes_I[:, 2]


# ============================================================
# CLEAN DATA
# ============================================================

good = (
    np.isfinite(u) &
    np.isfinite(v) &
    np.isfinite(real) &
    np.isfinite(imag) &
    np.isfinite(weight) &
    (weight > 0)
)

u = u[good]
v = v[good]
vis = real[good] + 1j * imag[good]
weight = weight[good]

print("Valid measurements:", len(vis))


# ============================================================
# UV COORDINATES
# ============================================================

frequency = float(header["CRVAL4"])

u *= frequency
v *= frequency

# UVFITS convention
u = -u
v = -v

baseline = np.hypot(u, v)

print("Frequency:", frequency / 1e9, "GHz")
print("Maximum baseline:", baseline.max() / 1e9, "Gλ")


# ============================================================
# HERMITIAN COMPLETION
# ============================================================

u = np.concatenate([u, -u])
v = np.concatenate([v, -v])
vis = np.concatenate([vis, np.conj(vis)])
weight = np.concatenate([weight, weight])

baseline = np.hypot(u, v)

short = baseline <= np.percentile(baseline, 5)
flux0 = np.median(np.abs(vis[short]))


# ============================================================
# IMAGE GRID
# ============================================================

pixel_rad = (
    FOV_UAS * RAD_PER_UAS /
    NPIX
)

axis = (
    np.arange(NPIX) -
    (NPIX - 1) / 2
) * pixel_rad

X, Y = np.meshgrid(axis, axis)
X = X.ravel()
Y = Y.ravel()

pixel_area = pixel_rad ** 2
n_pixels = NPIX ** 2


# ============================================================
# WEIGHTS + INITIAL FLUX
# ============================================================

weight = weight / np.median(weight)

short = baseline <= np.percentile(baseline, 5)

flux0 = np.median(np.abs(vis[short]))
flux0 = max(flux0, 1.0)

print("Initial flux:", flux0, "Jy")


# ============================================================
# FOURIER TRANSFORM
# ============================================================

def forward(image):

    image = image.ravel()
    result = np.empty(len(u), complex)

    for start in range(0, len(u), CHUNK):

        stop = min(start + CHUNK, len(u))

        phase = -2j * np.pi * (
            u[start:stop, None] * X +
            v[start:stop, None] * Y
        )

        result[start:stop] = (
            np.exp(phase) @ image
        ) * pixel_area

    return result


def adjoint(residual):

    result = np.zeros(n_pixels, complex)

    for start in range(0, len(u), CHUNK):

        stop = min(start + CHUNK, len(u))

        phase = 2j * np.pi * (
            u[start:stop, None] * X +
            v[start:stop, None] * Y
        )

        result += (
            np.exp(phase).T @ residual[start:stop]
        )

    return result * pixel_area


# ============================================================
# INITIAL IMAGE
# ============================================================

sigma = 15 * RAD_PER_UAS

gaussian = np.exp(
    -(X**2 + Y**2) /
    (2 * sigma**2)
)

gaussian /= gaussian.sum() * pixel_area

image0 = flux0 * gaussian


# ============================================================
# IMAGE PARAMETERIZATION
# ============================================================

def unpack(z):

    q = z[:-1]
    q -= q.max()

    p = np.exp(np.clip(q, -50, 50))
    p /= p.sum()

    flux = np.exp(np.clip(z[-1], -10, 10))

    image = flux * p / pixel_area

    return image, flux


z0 = np.r_[
    np.log(
        np.maximum(
            image0 * pixel_area,
            EPS
        )
    ),
    np.log(flux0)
]


# ============================================================
# REGULARIZATION
# ============================================================

def entropy(image):

    p = image * pixel_area
    p /= p.sum() + EPS
    p = np.clip(p, EPS, None)

    return np.sum(p * np.log(p))


def smoothness(image):

    image = image.reshape(NPIX, NPIX)

    dx = image[:, 1:] - image[:, :-1]
    dy = image[1:, :] - image[:-1, :]

    scale = image.mean() + EPS

    return (
        np.mean((dx / scale)**2) +
        np.mean((dy / scale)**2)
    )


# ============================================================
# OBJECTIVE
# ============================================================

weight_sum = weight.sum()


def objective(z):

    image, flux = unpack(z)

    model = forward(image)

    residual = model - vis

    chi2 = np.sum(
        weight *
        (
            residual.real**2 +
            residual.imag**2
        )
    ) / weight_sum

    value = (
        chi2 +
        ENTROPY_WEIGHT * entropy(image) +
        SMOOTHNESS_WEIGHT * smoothness(image)
    )

    # Gradient
    grad_image = (
        2 *
        np.real(
            adjoint(weight * residual)
        ) /
        weight_sum
    )

    p = np.exp(
        np.clip(
            z[:-1] - z[:-1].max(),
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

    mean_grad = np.sum(p * grad_pixel)

    grad_q = p * (
        grad_pixel - mean_grad
    )

    grad_flux = np.sum(
        grad_image * image
    )

    gradient = np.r_[
        grad_q,
        grad_flux
    ]

    return value, gradient


# ============================================================
# RECONSTRUCT
# ============================================================

print("Starting reconstruction...")

result = minimize(
    objective,
    z0,
    method="L-BFGS-B",
    jac=True,
    options={
        "maxiter": MAXITER,
        "ftol": 1e-9,
        "gtol": 1e-6
    }
)

print("Finished:", result.success)
print(result.message)


# ============================================================
# FINAL IMAGE
# ============================================================

image, recovered_flux = unpack(result.x)

image = image.reshape(NPIX, NPIX)

display = image / image.max()

extent = [
    axis[0] / RAD_PER_UAS,
    axis[-1] / RAD_PER_UAS,
    axis[0] / RAD_PER_UAS,
    axis[-1] / RAD_PER_UAS
]


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(8, 7))

plt.imshow(
    display,
    origin="lower",
    extent=extent,
    cmap="inferno"
)

plt.xlabel("Relative RA (μas)")
plt.ylabel("Relative Dec (μas)")
plt.title("M87* Reconstructed Image")

plt.colorbar(label="Normalized intensity")

plt.tight_layout()
plt.savefig(
    "m87_reconstructed.png",
    dpi=200
)

plt.show()

print("Recovered flux:", recovered_flux, "Jy")
print("Saved: m87_reconstructed.png")