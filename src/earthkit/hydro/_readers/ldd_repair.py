from typing import Literal, Optional, Tuple

import numpy as np


class LddRepair:
    """
    Repair LDD array - ensures all drainage paths end in a pit.
    Similar to pcraster's lddrepair function.

    The repair operation is done as follows:
    1. First, the cycles are removed by assigning missing values to all cells in a cycle.
    2. Second, cells with a local drain direction to the outside of the map or to a
       cell with a missing value (including cells that were in a cycle) are assigned
       the ldd code of a pit cell (code: 5, 255 or 247).
    3. Third, cells with a local drain direction to a cell with a missing value
       (including cells that were in a cycle) are assigned the ldd code of a pit cell (code: code: 5, 255 or 247).
    4. Fourth, cells with invalid LDD values (not in 1-9 or 1-128) are assigned the LDD code of a pit cell (code: 5)
       if any neighbor flows into it, otherwise they are set to missing value.
    """

    def __init__(
        self,
        ldd_array: np.ndarray,
        river_network_format: Literal["esri_d8", "pcr_d8", "merit_d8"] = "pcr_d8",
    ):
        self.ldd_array = ldd_array.copy()
        self.rows, self.cols = ldd_array.shape
        if river_network_format == "pcr_d8":
            self.__setup_pcr_d8()
        elif river_network_format == "esri_d8":
            self.__setup_esri_d8()
        elif river_network_format == "merit_d8":
            self.__setup_merit_d8()
        self.VALID_LDD_VALUES = set(self.LDD_OFFSETS.keys()).difference(
            {self.LDD_PIT_VALUE}
        )

    def repair(self) -> np.ndarray:
        """
        Repair LDD array - ensures all drainage paths end in a pit.
        Similar to pcraster's lddrepair function.

        The repair operation is done as follows:
        1. First, the cycles are removed by assigning missing values to all cells in a cycle.
        2. Second, cells with a local drain direction to the outside of the map or to a
        cell with a missing value (including cells that were in a cycle) are assigned
        the ldd code of a pit cell (code: 5, 255 or 247).
        3. Third, cells with a local drain direction to a cell with a missing value
        (including cells that were in a cycle) are assigned the ldd code of a pit cell (code: 5, 255 or 247).

        Returns:
        --------
        np.ndarray
            Repaired LDD array
        """

        # Step 1: Remove cycles by assigning missing values to all cells in a cycle
        # A cycle is a set of cells that don't drain to a pit because they drain to each other
        # We detect cycles by following downstream paths and checking for visited cells
        self.__remove_cycles()

        # Step 2: Find cells that are at the edge and flow out of bounds
        # These should be converted to pits
        self.__correct_edge_pits()

        # Step 2b: Find cells that drain to a cell with missing value (including cells in cycles)
        # These should be converted to pits
        self.__correct_invalid_downstream()

        # Step 3: For each invalid cell, check if any neighbor flows into it
        # If so, make this cell an outlet (pit = 5)
        self.__correct_invalid_values()

        return self.ldd_array

    def __setup_pcr_d8(self):
        """Set up constants and mappings for LDD repair."""

        self.LDD_PIT_VALUE = 5  # LDD value for pits (no flow)

        # LDD direction offsets for each direction value (1-9)
        # Format: (row_offset, col_offset)
        # Direction values follow PCRaster convention.
        self.LDD_OFFSETS = {
            9: (-1, 1),  # Northeast
            8: (-1, 0),  # North
            7: (-1, -1),  # Northwest
            6: (0, 1),  # East
            4: (0, -1),  # West
            3: (1, 1),  # Southeast
            2: (1, 0),  # South
            1: (1, -1),  # Southwest
            self.LDD_PIT_VALUE: (0, 0),  # Pit (no flow)
        }
        self.LDD_OFFSET_MIN = 1
        self.LDD_OFFSET_MAX = 9
        self.LDD_MISSING_VALUE = 0  # Value to use for missing/invalid LDD cells in the repaired LDD

    def __setup_esri_d8(self):
        """Set up constants and mappings for LDD repair."""

        self.LDD_PIT_VALUE = 255  # LDD value for pits (no flow)

        # LDD direction offsets for each direction value (1-128)
        # Format: (row_offset, col_offset)
        # Direction values follow esri_d8 convention
        self.LDD_OFFSETS = {
            128: (-1, 1),  # Northeast
            64: (-1, 0),  # North
            32: (-1, -1),  # Northwest
            1: (0, 1),  # East
            16: (0, -1),  # West
            2: (1, 1),  # Southeast
            4: (1, 0),  # South
            8: (1, -1),  # Southwest
            self.LDD_PIT_VALUE: (0, 0),  # Pit (no flow)
        }
        self.LDD_OFFSET_MIN = 1
        self.LDD_OFFSET_MAX = 128
        self.LDD_MISSING_VALUE = -1  # Value to use for missing/invalid LDD cells in the repaired LDD (ESRI uses -1)

    def __setup_merit_d8(self):
        """Set up constants and mappings for LDD repair."""
        self.__setup_esri_d8()  # MERIT d8 uses the same direction values as ESRI d8, but with a different pit value
        esri_d8_pit_value = self.LDD_PIT_VALUE
        self.LDD_PIT_VALUE = 247  # LDD value for pits (no flow)
        self.LDD_OFFSETS[self.LDD_PIT_VALUE] = self.LDD_OFFSETS.pop(
            esri_d8_pit_value
        )  # Pit (no flow)

    def __get_downstream(self, i: int, j: int) -> Optional[Tuple[int, int]]:
        """
        Get the downstream cell coordinates for a given cell.
        Parameters:
        -----------
        i : int
            Row index of the cell.
        j : int
            Column index of the cell.

        Returns:
        --------
        Optional[Tuple[int, int]]
            Coordinates of the downstream cell, or None if no valid downstream cell exists.
        """
        ldd_val = int(self.ldd_array[i, j])
        if self.__is_pit_or_invalid(ldd_val):
            return None  # Invalid or pit
        di, dj = self.LDD_OFFSETS[ldd_val]
        next_i, next_j = i + di, j + dj
        if 0 <= next_i < self.rows and 0 <= next_j < self.cols:
            return (next_i, next_j)
        return None  # Flows out of bounds

    def __remove_cycles(self):
        """
        Remove cycles from the LDD array by assigning missing values to all cells in a cycle.
        A cycle is a set of cells that don't drain to a pit because they drain to each other.
        We detect cycles by following downstream paths and checking for visited cells.
        """
        # Track cells that are part of cycles
        cycle_cells = set()

        # For each cell, follow the downstream path to detect cycles
        for i in range(self.rows):
            for j in range(self.cols):
                ldd_val = int(self.ldd_array[i, j])

                # Skip invalid cells and pits
                if self.__is_pit_or_invalid(ldd_val):
                    continue

                # Follow the downstream path, tracking visited cells
                path = []  # List of (i, j) tuples in order visited
                visited_in_path = set()  # Set of cells visited in current path

                current_i, current_j = i, j

                while current_i is not None and current_j is not None:
                    # Check if we've hit a pit or invalid cell - no cycle here
                    current_ldd = int(self.ldd_array[current_i, current_j])
                    if self.__is_pit_or_invalid(current_ldd):
                        break

                    # Check if we've already visited this cell in the current path - cycle detected!
                    if (current_i, current_j) in visited_in_path:
                        # Found a cycle - mark all cells in the cycle as missing
                        # Find the start of the cycle in the path
                        cycle_start_idx = path.index((current_i, current_j))
                        for cycle_i, cycle_j in path[cycle_start_idx:]:
                            cycle_cells.add((cycle_i, cycle_j))
                        break

                    # Add current cell to path
                    path.append((current_i, current_j))
                    visited_in_path.add((current_i, current_j))

                    # Move to downstream cell
                    downstream = self.__get_downstream(current_i, current_j)
                    if downstream is None:
                        break
                    current_i, current_j = downstream

        # Assign missing values to all cells in cycles
        for i, j in cycle_cells:
            self.ldd_array[i, j] = self.LDD_MISSING_VALUE

    def __is_pit_or_invalid(self, ldd_val: int) -> bool:
        """
        Check if the LDD value is a pit or invalid.
        Parameters:
        -----------
        ldd_val : int
            LDD value to check.
        Returns:
        --------
            bool
                True if the LDD value is a pit or invalid, False otherwise.
        """
        return (ldd_val == self.LDD_PIT_VALUE or
                ldd_val < self.LDD_OFFSET_MIN or
                ldd_val > self.LDD_OFFSET_MAX)

    def __correct_edge_pits(self):
        """
        Correct edge cells that flow out of bounds by assigning them
        the LDD code of a pit cell (5).
        """
        for i in range(self.rows):
            for j in range(self.cols):
                ldd_val = int(self.ldd_array[i, j])

                # Skip if already a pit (5) or invalid
                if self.__is_pit_or_invalid(ldd_val):
                    continue

                # Check if this cell's immediate downstream is out of bounds
                di, dj = self.LDD_OFFSETS[ldd_val]
                next_i, next_j = i + di, j + dj

                if not (0 <= next_i < self.rows and 0 <= next_j < self.cols):
                    # Flows out of bounds - convert to pit
                    self.ldd_array[i, j] = self.LDD_PIT_VALUE

    def __correct_invalid_downstream(self):
        """
        Correct cells that flow to a cell with missing value (including cells
        in cycles) by assigning them the LDD code of a pit cell (5).
        """
        for i in range(self.rows):
            for j in range(self.cols):
                ldd_val = int(self.ldd_array[i, j])

                # Skip if already a pit (5) or invalid
                if self.__is_pit_or_invalid(ldd_val):
                    continue

                # Check if this cell's downstream is a missing value
                di, dj = self.LDD_OFFSETS[ldd_val]
                next_i, next_j = i + di, j + dj

                if 0 <= next_i < self.rows and 0 <= next_j < self.cols:
                    downstream_val = self.ldd_array[next_i, next_j]
                    if downstream_val == self.LDD_MISSING_VALUE:
                        # Drains to a missing value - convert to pit
                        self.ldd_array[i, j] = self.LDD_PIT_VALUE

    def __correct_invalid_values(self):
        """
        Correct cells with invalid LDD values (not in 1-9) by assigning them the LDD code of a pit cell (5).
        """
        # Find invalid values (not 1-9)
        invalid_mask = ((self.ldd_array < self.LDD_OFFSET_MIN) |
                        (self.ldd_array > self.LDD_OFFSET_MAX)
                    ) & (self.ldd_array != self.LDD_MISSING_VALUE)
        for i in range(self.rows):
            for j in range(self.cols):
                if invalid_mask[i, j]:
                    # Check if any neighbor flows into this cell
                    is_outlet = False
                    for direction_value in self.VALID_LDD_VALUES:
                        (di, dj) = self.LDD_OFFSETS[direction_value]
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.rows and 0 <= nj < self.cols:
                            # Check if neighbor flows into this cell
                            # The neighbor's LDD should point to this cell
                            neighbor_ldd = self.ldd_array[ni, nj]
                            if neighbor_ldd in self.VALID_LDD_VALUES:
                                # Check if the direction points to current cell
                                ndi, ndj = self.LDD_OFFSETS[neighbor_ldd]
                                if ni + ndi == i and nj + ndj == j:
                                    is_outlet = True
                                    break
                    # If any neighbor flows into this cell, make it a pit (5), otherwise set to missing value
                    self.ldd_array[i, j] = (
                        self.LDD_PIT_VALUE if is_outlet else self.LDD_MISSING_VALUE
                    )
