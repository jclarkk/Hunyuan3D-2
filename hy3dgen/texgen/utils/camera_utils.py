from math import sin, cos, atan2, asin, sqrt, pi, degrees


def _fibonacci_sphere(num_samples=12):
    points = []
    phi = pi * (3. - sqrt(5.))
    for i in range(num_samples):
        y = 1 - (i / float(num_samples - 1)) * 2
        radius = sqrt(1 - y * y)
        theta = phi * i
        x = cos(theta) * radius
        z = sin(theta) * radius
        points.append((x, y, z))
    return points


def _cartesian_to_spherical(x, y, z):
    azim = degrees(atan2(z, x))
    r = sqrt(x * x + y * y + z * z)
    elev = degrees(asin(y / r))
    return azim, elev


def dynamic_camera_angles(num_views=12, enhanced=False):
    points = _fibonacci_sphere(num_views)
    azims, elevs, weights, camera_indices = [], [], [], []

    for x, y, z in points:
        azim, elev = _cartesian_to_spherical(x, y, z)
        azims.append(azim)
        elevs.append(elev)

        # Generate appropriate weights (higher for front-facing and key views)
        # Higher weight for front view (azim near 0, elev near 0)
        weight = 0.1
        if abs(azim) < 30 and abs(elev) < 30:
            weight = 1.0  # Front view gets higher weight
        elif abs(azim - 180) < 30 and abs(elev) < 30:
            weight = 0.5  # Back view gets medium weight
        weights.append(weight)

        # Calculate camera_info index using similar logic to original code
        # First, quantize elevation to nearest standard value
        if elev < -75:
            quantized_elev = -90
        elif elev < -30:
            quantized_elev = -45
        elif elev < -17.5:
            quantized_elev = -20
        elif elev < -7.5:
            quantized_elev = -15
        elif elev < 7.5:
            quantized_elev = 0
        elif elev < 17.5:
            quantized_elev = 15
        elif elev < 30:
            quantized_elev = 20
        elif elev < 75:
            quantized_elev = 45
        else:
            quantized_elev = 90

        # Lookup tables similar to original code
        if enhanced:
            elev_divisor = {
                -90: 3, -45: 3, -20: 1, -15: 1, 0: 1, 15: 1, 20: 1, 45: 2, 90: 3
            }
            elev_offset = {
                -90: 36, -45: 36, -20: 0, -15: 0, 0: 12, 15: 24, 20: 24, 45: 30, 90: 40
            }
        else:
            elev_divisor = {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}
            elev_offset = {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}

        # Calculate camera index similar to original formula
        if quantized_elev in elev_divisor and quantized_elev in elev_offset:
            camera_index = (((int(azim) // 30) + 9) % 12) // elev_divisor[quantized_elev] + elev_offset[quantized_elev]
        else:
            # Fallback for elevations not in the lookup tables
            camera_index = (((int(azim) // 30) + 9) % 12) // 1 + 12  # Default to mid-level

        camera_indices.append(camera_index)

    return azims, elevs, weights, camera_indices
