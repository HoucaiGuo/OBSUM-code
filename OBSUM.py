import numpy as np
from scipy.optimize import lsq_linear
from skimage.transform import downscale_local_mean, resize


class OBSUM:
    def __init__(self, F_tb, C_tb, C_tp, F_tb_class, F_tb_objects,
                 class_num=5, scale_factor=30, win_size=15,
                 OL_RC_percent=5,
                 similar_win_size=31, similar_num=30,
                 min_val=0.0, max_val=1.0):
        self.F_tb = F_tb.astype(np.float32)
        self.C_tb = C_tb.astype(np.float32)
        self.C_tp = C_tp.astype(np.float32)
        self.delta_C = self.C_tp - self.C_tb
        self.F_tb_class = F_tb_class
        self.F_tb_objects = F_tb_objects
        self.class_num = class_num
        self.scale_factor = scale_factor
        self.win_size = win_size
        self.OL_RC_percent = OL_RC_percent
        self.similar_win_size = similar_win_size
        self.similar_num = similar_num
        self.min_val = min_val
        self.max_val = max_val

    def refine_classification_using_objects(self):
        """
        Refine the land-cover classification map using the segmented image objects.
        """
        refined_class = np.empty(shape=self.F_tb_class.shape, dtype=np.uint8)

        object_indices = np.unique(self.F_tb_objects)
        for object_idx in object_indices:
            object_mask = self.F_tb_objects == object_idx
            object_classes = self.F_tb_class[object_mask]
            if np.count_nonzero(object_mask) == 1:
                object_class = object_classes
            else:
                object_class = np.argmax(np.bincount(object_classes.squeeze()))
            refined_class[object_mask] = object_class

        self.F_tb_class = refined_class
        print(f"Refined land-cover classification map!")

    def calculate_class_fractions(self):
        """
        Calculate the fractions of the land-cover classes inside each coarse pixel.

        Returns
        -------
        C_fractions : array_like, (C_row, C_col, class_num) shaped
            Fractions of the land cover classes.
        """
        C_fractions = np.zeros(shape=(self.C_tp.shape[0], self.C_tp.shape[1], self.class_num), dtype=np.float32)
        for row_idx in range(self.C_tp.shape[0]):
            for col_idx in range(self.C_tp.shape[1]):
                F_class_pixels = self.F_tb_class[row_idx * self.scale_factor:(row_idx + 1) * self.scale_factor,
                                 col_idx * self.scale_factor:(col_idx + 1) * self.scale_factor]
                for class_idx in range(self.class_num):
                    pixel_num = np.count_nonzero(F_class_pixels == class_idx)
                    C_fractions[row_idx, col_idx, class_idx] = pixel_num / (self.scale_factor * self.scale_factor)

        return C_fractions

    def unmix_window(self, C_values, C_fractions, lower_bound, upper_bound):
        """
        Use a constrained least squares method to unmix the coarse pixel to get the reflectance of the each class
        in the predicted image covered by the central coarse pixel of the moving window.

        Parameters
        ----------
        C_values : array_like, (win_size**2, 1) shaped
            Reflectances of the coarse pixels in the moving window.
        C_fractions : array_like, (win_size**2, class_num) shaped
            Fractions of all the classes inside each coarse pixel.
        lower_bound : float
            Lower bound in lsq estimation.
        lower_bound : float
            Upper bound in lsq estimation.
        Returns
        -------
        result : array_like, (class_num, 1) shaped
            The unmixed changes.
        """
        lsq = lsq_linear(C_fractions, C_values,
                         bounds=(lower_bound, upper_bound), method="bvls", max_iter=100)

        result = lsq.x

        return result

    def calculate_distances_in_coarse_pixel(self):
        """
        For each fine pixel within a coarse pixel, calculate its distance to the central fine pixel.
        """
        rows = np.linspace(start=0, stop=self.scale_factor - 1, num=self.scale_factor)
        cols = np.linspace(start=0, stop=self.scale_factor - 1, num=self.scale_factor)
        xx, yy = np.meshgrid(rows, cols, indexing='ij')

        central_row = self.scale_factor // 2
        central_col = self.scale_factor // 2
        distances = np.sqrt(np.square(xx - central_row) + np.square(yy - central_col))

        distances = np.concatenate([distances for i in range(self.C_tp.shape[0])], axis=0)
        distances = np.concatenate([distances for i in range(self.C_tp.shape[1])], axis=1)

        # normalize to [1, 1+sqrt(2)]
        distances = 1 + distances / (self.scale_factor // 2)

        return distances

    def calculate_object_homogeneity_index(self):
        """
        For each fine pixel, calculate the object homogeneity index inside the local window (size is one coarse pixel).
        """
        object_homo_index = np.zeros(shape=(self.F_tb.shape[0], self.F_tb.shape[1]), dtype=np.float32)
        F_tb_objects_pad = np.pad(self.F_tb_objects, pad_width=((self.scale_factor // 2, self.scale_factor // 2),
                                                                (self.scale_factor // 2, self.scale_factor // 2)),
                                  mode="reflect")
        for row_idx in range(self.F_tb.shape[0]):
            for col_idx in range(self.F_tb.shape[1]):
                current_object = self.F_tb_objects[row_idx, col_idx]
                pixel_objects = F_tb_objects_pad[row_idx:row_idx + self.scale_factor,
                                col_idx:col_idx + self.scale_factor]
                object_homo_index[row_idx, col_idx] = np.count_nonzero(pixel_objects == current_object) / \
                                                      np.square(self.scale_factor)

        return object_homo_index

    def calculate_similar_pixel_distances(self):
        """
        Calculate similar pixels' distances to the central target pixel.
        """
        rows = np.linspace(start=0, stop=self.similar_win_size - 1, num=self.similar_win_size)
        cols = np.linspace(start=0, stop=self.similar_win_size - 1, num=self.similar_win_size)
        xx, yy = np.meshgrid(rows, cols, indexing='ij')

        central_row = self.similar_win_size // 2
        central_col = self.similar_win_size // 2
        distances = np.sqrt(np.square(xx - central_row) + np.square(yy - central_col))

        # normalize to [1, 1+sqrt(2)]
        distances = 1 + distances / (self.similar_win_size // 2)

        return distances

    def select_similar_pixels(self):
        """
        Select similar pixels for pixel-level residual compensation.
        """
        F_tb_pad = np.pad(self.F_tb,
                          pad_width=((self.similar_win_size // 2, self.similar_win_size // 2),
                                     (self.similar_win_size // 2, self.similar_win_size // 2),
                                     (0, 0)),
                          mode="reflect")
        F_tb_similar_weights = np.empty(shape=(self.F_tb.shape[0], self.F_tb.shape[1], self.similar_num),
                                        dtype=np.float32)
        F_tb_similar_indices = np.empty(shape=(self.F_tb.shape[0], self.F_tb.shape[1], self.similar_num),
                                        dtype=np.uint32)

        distances = self.calculate_similar_pixel_distances().flatten()
        for row_idx in range(self.F_tb.shape[0]):
            for col_idx in range(self.F_tb.shape[1]):
                central_pixel_vals = self.F_tb[row_idx, col_idx, :]
                neighbor_pixel_vals = F_tb_pad[row_idx:row_idx + self.similar_win_size,
                                      col_idx:col_idx + self.similar_win_size, :]
                D = np.mean(np.abs(neighbor_pixel_vals - central_pixel_vals), axis=2).flatten()
                similar_indices = np.argsort(D)[:self.similar_num]
                similar_distances = 1 + distances[similar_indices] / (self.similar_win_size // 2)
                similar_weights = (1 / similar_distances) / np.sum(1 / similar_distances)

                F_tb_similar_indices[row_idx, col_idx, :] = similar_indices
                F_tb_similar_weights[row_idx, col_idx, :] = similar_weights

        return F_tb_similar_indices, F_tb_similar_weights

    def object_based_spatial_unmixing(self):
        """
        Object-Based Spatial Unmixing Model.

        Returns
        -------
        F_tp_OBSUM : array_like
            The predicted fine image at tp.
        """
        ###########################################################
        #                      Initialization                     #
        ###########################################################
        F_tp_OBSUM = np.empty(shape=(self.F_tb.shape[0], self.F_tb.shape[1], self.C_tp.shape[2]),
                              dtype=self.C_tp.dtype)

        # refine the image classes using image objects
        self.refine_classification_using_objects()

        # calculate the class fractions
        C_fractions = self.calculate_class_fractions()

        # calculate the object residual index which is used to perform the object-level residual compensation (OL-RC)
        distances_in_C = self.calculate_distances_in_coarse_pixel()
        object_homo_index = self.calculate_object_homogeneity_index()
        object_residual_index = object_homo_index / distances_in_C

        # pad the coarse temporal change and the fraction maps since the unmixing process is based on a local window
        delta_C_pad = np.pad(self.delta_C, pad_width=((self.win_size // 2, self.win_size // 2),
                                                      (self.win_size // 2, self.win_size // 2), (0, 0)), mode="reflect")
        C_fractions_pad = np.pad(C_fractions, pad_width=((self.win_size // 2, self.win_size // 2),
                                                         (self.win_size // 2, self.win_size // 2), (0, 0)),
                                 mode="reflect")
        object_indices = np.unique(self.F_tb_objects)

        # select similar pixels for pixel level residual compensation (PL-RC)
        F_tb_similar_indices, F_tb_similar_weights = self.select_similar_pixels()
        print("Selected similar pixels!")

        ####################################################################
        #    Apply the Object-Based Spatial Unmixing Model band-by-band    #
        ####################################################################
        for band_idx in range(self.C_tp.shape[2]):
            lower_bound = np.min(delta_C_pad[:, :, band_idx])
            upper_bound = np.max(delta_C_pad[:, :, band_idx])
            SU_prediction = np.empty(shape=(self.F_tb.shape[0], self.F_tb.shape[1]), dtype=self.C_tp.dtype)
            ###########################################################
            #              1. Object-level unmxing (OL-U)             #
            ###########################################################
            for row_idx in range(self.C_tp.shape[0]):
                for col_idx in range(self.C_tp.shape[1]):
                    C_pixels_win = delta_C_pad[row_idx:row_idx + self.win_size, col_idx:col_idx + self.win_size,
                                   band_idx]
                    C_fractions_win = C_fractions_pad[row_idx:row_idx + self.win_size,
                                      col_idx:col_idx + self.win_size, :]

                    # unmix the local window
                    F_values = self.unmix_window(C_pixels_win.flatten(),
                                                 C_fractions_win.reshape(C_pixels_win.shape[0] * C_pixels_win.shape[1],
                                                                         self.class_num),
                                                 lower_bound, upper_bound)

                    # class information of the central coarse pixel of the window
                    C_classes = self.F_tb_class[row_idx * self.scale_factor:(row_idx + 1) * self.scale_factor,
                                col_idx * self.scale_factor:(col_idx + 1) * self.scale_factor]

                    # assign the reflectances of the fine pixels inside current coarse pixel class-by-class
                    for class_idx in range(self.class_num):
                        class_mask = C_classes == class_idx
                        SU_prediction[row_idx * self.scale_factor:(row_idx + 1) * self.scale_factor,
                        col_idx * self.scale_factor:(col_idx + 1) * self.scale_factor][class_mask] = \
                            F_values[class_idx]

            for object_idx in object_indices:
                # assign the unmixed value
                object_mask = self.F_tb_objects == object_idx
                object_value = np.mean(SU_prediction[object_mask])
                F_tp_OBSUM[:, :, band_idx][object_mask] = self.F_tb[:, :, band_idx][object_mask] + object_value
            print(f"Finished initial prediction of band {band_idx}!")
            F_tp_OBSUM[F_tp_OBSUM > self.max_val] = self.max_val
            F_tp_OBSUM[F_tp_OBSUM < self.min_val] = self.min_val

            ###########################################################
            #      2. Object-level residual compensation (OL-RC)      #
            ###########################################################
            # calculate coarse residuals and downscale them to fine scale using bi-cubic interpolation
            C_tp_prediction = downscale_local_mean(F_tp_OBSUM[:, :, band_idx],
                                                   factors=(self.scale_factor, self.scale_factor))
            C_residuals = self.C_tp[:, :, band_idx] - C_tp_prediction
            F_residuals = resize(C_residuals, output_shape=(self.F_tb.shape[0], self.F_tb.shape[1]), order=3)

            for object_idx in object_indices:
                object_mask = self.F_tb_objects == object_idx
                object_residuals = F_residuals[object_mask]

                # residual selection
                residual_indices = object_residual_index[object_mask]
                indices = (residual_indices >= np.percentile(residual_indices, 100 - self.OL_RC_percent)).nonzero()[0]
                selected_residuals = object_residuals[indices]

                # use weighted residuals
                selected_weights = residual_indices[indices]
                selected_weights = selected_weights / np.sum(selected_weights)
                residual = np.sum(selected_weights * selected_residuals)

                # assign the predicted residual
                F_tp_OBSUM[:, :, band_idx][object_mask] += residual
            print(f"Finished object-level residual compensation of band {band_idx}!")
            F_tp_OBSUM[F_tp_OBSUM > self.max_val] = self.max_val
            F_tp_OBSUM[F_tp_OBSUM < self.min_val] = self.min_val

            ###########################################################
            #      3. Pixel-level residual compensation (PL-RC)       #
            ###########################################################
            C_tp_prediction = downscale_local_mean(F_tp_OBSUM[:, :, band_idx],
                                                   factors=(self.scale_factor, self.scale_factor))
            C_residuals = self.C_tp[:, :, band_idx] - C_tp_prediction
            F_residuals = resize(C_residuals, output_shape=(self.F_tb.shape[0], self.F_tb.shape[1]), order=3)
            F_residuals_pad = np.pad(F_residuals,
                                     pad_width=((self.similar_win_size // 2, self.similar_win_size // 2),
                                                (self.similar_win_size // 2, self.similar_win_size // 2)),
                                     mode="reflect")
            for row_idx in range(F_residuals.shape[0]):
                for col_idx in range(F_residuals.shape[1]):
                    neighbor_pixel_residuals = F_residuals_pad[row_idx:row_idx + self.similar_win_size,
                                               col_idx:col_idx + self.similar_win_size]
                    # similar pixels
                    similar_indices = F_tb_similar_indices[row_idx, col_idx, :]
                    similar_residuals = neighbor_pixel_residuals.flatten()[similar_indices]
                    similar_weights = F_tb_similar_weights[row_idx, col_idx, :]

                    # use weighted residuals
                    residual = np.sum(similar_residuals * similar_weights)

                    # assign the predicted residual
                    F_tp_OBSUM[row_idx, col_idx, band_idx] += residual
            print(f"Finished final prediction of band {band_idx}!")
            F_tp_OBSUM[F_tp_OBSUM > self.max_val] = self.max_val
            F_tp_OBSUM[F_tp_OBSUM < self.min_val] = self.min_val

        return F_tp_OBSUM
