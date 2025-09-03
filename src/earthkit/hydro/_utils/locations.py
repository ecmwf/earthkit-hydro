import numpy as np


def locations_to_1d(xp, river_network, locations):

    orig_locations = locations
    dict_locations = isinstance(locations, dict)
    if dict_locations:

        coord1_network_vals, coord2_network_vals = river_network.coords.values()

        locations = []
        if river_network.shape is None:  # vector network
            for coord1_val, coord2_val in orig_locations.values():
                indx = (
                    (coord1_val - coord1_network_vals) ** 2
                    + (coord2_val - coord2_network_vals) ** 2
                ).argmin()
                locations.append(int(indx))
        else:
            for coord1_val, coord2_val in orig_locations.values():
                indx = np.argmin((coord1_val - coord1_network_vals) ** 2)
                indy = np.argmin((coord2_val - coord2_network_vals) ** 2)
                locations.append((int(indx), int(indy)))

    locations = xp.asarray(locations, device=river_network.device)
    stations = locations

    if stations.ndim == 2 and stations.shape[1] == 2:
        if xp.name not in ["numpy", "cupy"]:
            raise NotImplementedError
        # TODO: make this code actually xp agnostic
        rows, cols = stations[:, 0], stations[:, 1]
        flat_indices = rows * river_network.shape[1] + cols
        flat_mask = river_network.mask
        reverse_map = -xp.ones(
            river_network.shape[0] * river_network.shape[1], dtype=int
        )
        reverse_map[flat_mask] = xp.arange(flat_mask.shape[0])
        masked_indices = reverse_map[flat_indices]
        if xp.any(masked_indices < 0):
            raise ValueError(
                "Some station points are not included in the masked array."
            )
        stations = xp.asarray(masked_indices, device=river_network.device)
    else:
        assert stations.ndim == 1

    return stations, locations, orig_locations
