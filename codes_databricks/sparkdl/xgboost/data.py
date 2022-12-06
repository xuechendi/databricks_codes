#
# DATABRICKS CONFIDENTIAL & PROPRIETARY
# __________________
#
# Copyright 2020 Databricks, Inc.
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property of Databricks, Inc.
# and its suppliers, if any.  The intellectual and technical concepts contained herein are
# proprietary to Databricks, Inc. and its suppliers and may be covered by U.S. and foreign Patents,
# patents in process, and are protected by trade secret and/or copyright law. Dissemination, use,
# or reproduction of this information is strictly forbidden unless prior written permission is
# obtained from Databricks, Inc.
#
# If you view or obtain a copy of this information and believe Databricks, Inc. may not have
# intended it to be made available, please promptly report it to Databricks Legal Department
# @ legal@databricks.com.
#

# pylint: disable=invalid-name
# pylint: disable=R0912

import numpy as np
from scipy.sparse import csr_matrix
from xgboost import DMatrix


# Since sklearn's SVM converter doesn't address weights, this one does address weights:
def _dump_libsvm(features, labels, weights=None, external_storage_precision=5):
    esp = external_storage_precision
    lines = []

    def gen_label_str(row_idx):
        if weights is not None:
            return "{label:.{esp}g}:{weight:.{esp}g}".format(
                label=labels[row_idx], esp=esp, weight=weights[row_idx])
        else:
            return "{label:.{esp}g}".format(label=labels[row_idx], esp=esp)

    def gen_feature_value_str(feature_idx, feature_val):
        return "{idx:.{esp}g}:{value:.{esp}g}".format(
            idx=feature_idx, esp=esp, value=feature_val
        )

    is_csr_matrix = isinstance(features, csr_matrix)

    for i in range(len(labels)):
        current = [gen_label_str(i)]
        if is_csr_matrix:
            idx_start = features.indptr[i]
            idx_end = features.indptr[i + 1]
            for idx in range(idx_start, idx_end):
                j = features.indices[idx]
                val = features.data[idx]
                current.append(gen_feature_value_str(j, val))
        else:
            for j, val in enumerate(features[i]):
                current.append(gen_feature_value_str(j, val))
        lines.append(" ".join(current) + "\n")
    return lines


# This is the updated version that handles weights
def _stream_train_val_data(features, labels, weights, main_file,
                           external_storage_precision):
    lines = _dump_libsvm(features, labels, weights, external_storage_precision)
    main_file.writelines(lines)


def _stream_data_into_libsvm_file(data_iterator, has_weight, missing,
                                  has_validation, file_prefix,
                                  external_storage_precision):
    # getting the file names for storage
    train_file_name = file_prefix + "/data.txt.train"
    train_file = open(train_file_name, "w")
    if has_validation:
        validation_file_name = file_prefix + "/data.txt.val"
        validation_file = open(validation_file_name, "w")

    # iterating through each pdf in the pandas data_iterator
    for pdf in data_iterator:

        def gen_row_tuple(pdf):
            for tup in pdf.itertuples(index=False):
                yield tup

        train_val_data = _process_row_tuple_iter(gen_row_tuple(pdf),
                                                 train=True,
                                                 has_weight=has_weight,
                                                 missing=missing,
                                                 has_validation=has_validation)
        if has_validation:
            train_X, train_y, train_w, _, val_X, val_y, val_w, _ = train_val_data
            _stream_train_val_data(train_X, train_y, train_w, train_file,
                                   external_storage_precision)
            _stream_train_val_data(val_X, val_y, val_w, validation_file,
                                   external_storage_precision)
        else:
            train_X, train_y, train_w, _ = train_val_data
            _stream_train_val_data(train_X, train_y, train_w, train_file,
                                   external_storage_precision)

    if has_validation:
        train_file.close()
        validation_file.close()
        return train_file_name, validation_file_name
    else:
        train_file.close()
        return train_file_name


def _create_dmatrix_from_file(file_path, cache_file_path, dmatrix_kwargs):
    return DMatrix(file_path, **dmatrix_kwargs)


def prepare_train_val_data(data_iterator,
                           has_weight,
                           missing,
                           has_validation,
                           has_fit_base_margin=False):
    def gen_row_tuple():
        for pdf in data_iterator:
            for tup in pdf.itertuples(index=False):
                yield tup

    return _process_row_tuple_iter(gen_row_tuple(),
                                   train=True,
                                   has_weight=has_weight,
                                   missing=missing,
                                   has_validation=has_validation,
                                   has_fit_base_margin=has_fit_base_margin,
                                   has_predict_base_margin=False)


def prepare_predict_data(ftype, fsize, findices, fvalues, missing,
                         base_margin):
    # Generate basic tuple instead of namedtuple for better performance
    if str(type(base_margin)) != "<class 'NoneType'>":
        row_tuple_iter = zip(iter(ftype), iter(fsize), iter(findices),
                             iter(fvalues), iter(base_margin))
        return _process_row_tuple_iter(row_tuple_iter,
                                       train=False,
                                       has_weight=False,
                                       missing=missing,
                                       has_validation=False,
                                       has_fit_base_margin=False,
                                       has_predict_base_margin=True)
    else:
        row_tuple_iter = zip(iter(ftype), iter(fsize), iter(findices),
                             iter(fvalues))
        return _process_row_tuple_iter(row_tuple_iter,
                                       train=False,
                                       has_weight=False,
                                       missing=missing,
                                       has_validation=False,
                                       has_fit_base_margin=False,
                                       has_predict_base_margin=False)


def _check_feature_dims(num_dims, expected_dims):
    """
    Check all feature vectors has the same dimension
    """
    if expected_dims is None:
        return num_dims
    if num_dims != expected_dims:
        raise ValueError("Rows contain different feature dimensions: "
                         "Expecting {}, got {}.".format(
                             expected_dims, num_dims))
    return expected_dims


def _unwrap_row_tuple(tup, train, has_weight, has_fit_base_margin,
                      has_predict_base_margin):
    label, weight, base_margin = None, None, None
    # Use index getter for better performance
    # Row format:
    #  first 4 properties: type, size, indices, values
    #   (the same with fields in spark VectorUDT type struct)
    #  with optional label, weight, validationIndicator properties appened at the end
    # If only include the first 4 properties, the tup can be basic tuple type,
    # otherwise it must be namedtuple type
    ftype, fsize, findices, fvalues = tup[0], tup[1], tup[2], tup[3]
    if train:
        label = tup.label
        if has_weight:
            weight = tup.weight
        if has_fit_base_margin:
            base_margin = tup.baseMargin
    if not train and has_predict_base_margin:
        base_margin = tup[4]
    return ftype, fsize, findices, fvalues, label, weight, base_margin


def _extract_from_sparse_or_dense_vector(ftype, fsize, findices, fvalues):
    """
    Extract the number of feature dimensions and csr indices from unwrapped sparse vector
    or dense vector. Note that for dense vector, fsize and findices are None, so we need to
    calculate them from fvalues.
    """
    if ftype == 0:  # sparse vector
        return int(fsize), findices
    # dense vector
    return len(fvalues), np.array(range(len(fvalues)))


def _row_tuple_list_to_feature_matrix_y_w(row_tuple_list, train, has_weight,
                                          has_fit_base_margin,
                                          has_predict_base_margin):
    """
    Construct a feature matrix in csr_matrix format, label array y and weight array w
    from the row_tuple_list.
    If train == False, y and w will be None.
    If has_weight == False, w will be None.
    If has_base_margin == False, b_m will be None.
    Note: the row_tuple_list will be cleared during
    executing for reducing peak memory consumption
    """
    expected_feature_dims = None
    label_list, weight_list, base_margin_list = [], [], []
    # variables for constructing csr_matrix
    indices_list, indptr_list, values_list = [], [0], []

    # Process rows
    for tup in row_tuple_list:
        ftype, fsize, findices, fvalues, label, weight, base_margin = _unwrap_row_tuple(
            tup, train, has_weight, has_fit_base_margin,
            has_predict_base_margin)
        num_feature_dims, csr_indices = _extract_from_sparse_or_dense_vector(
            ftype, fsize, findices, fvalues)

        expected_feature_dims = _check_feature_dims(num_feature_dims,
                                                    expected_feature_dims)

        indices_list.append(csr_indices)
        indptr_list.append(indptr_list[-1] + len(csr_indices))
        values_list.append(fvalues)
        if train:
            label_list.append(label)
        if has_weight:
            weight_list.append(weight)
        if has_fit_base_margin or has_predict_base_margin:
            base_margin_list.append(base_margin)

    # clear input list for reducing peak memory consumption
    row_tuple_list.clear()

    # Construct feature_matrix
    if expected_feature_dims is None:
        raise ValueError("The input dataframe is empty!")

    indptr_arr = np.array(indptr_list)
    indices_arr = np.concatenate(indices_list)
    values_arr = np.concatenate(values_list)
    num_rows = len(values_list)
    feature_matrix = csr_matrix((values_arr, indices_arr, indptr_arr),
                                shape=(num_rows, expected_feature_dims))

    # Construct y and w
    y = np.array(label_list, dtype=np.float) if train else None
    w = np.array(weight_list, dtype=np.float) if has_weight else None
    b_m = np.array(base_margin_list, dtype=np.float) if (
        has_fit_base_margin or has_predict_base_margin) else None
    return feature_matrix, y, w, b_m


def _handle_missing_values(feature_matrix, missing):
    if missing == 0:
        # If missing value is zero, we make all 0 entries inactive and pass the csr_matrix to
        # DMatrix, since DMatrix will treat all inactive elements in the csr_matrix as missing.
        feature_matrix.eliminate_zeros()
        X = feature_matrix
    else:
        # Convert features_matrix to be dense and DMatrix will respect missing value
        # setting.
        # TODO: Use csr sparse matrix to optimize this.
        X = feature_matrix.toarray()
    return X


def _process_row_tuple_list(row_tuple_list, train, has_weight, missing,
                            has_fit_base_margin, has_predict_base_margin):
    """
    Collect data from a list of row tuples to the array-like matrices
    feature matrix X, label array y and weight array w.
    X guarantees consistent missing value semantics in xgboost and spark.
    If train == False, y and w will be None.
    If has_weight == False, w will be None.
    If has_base_margin == False, b_m will be None.
    Note: the row_tuple_list will be cleared during executing for reducing peak memory consumption
    """
    feature_matrix, y, w, b_m = _row_tuple_list_to_feature_matrix_y_w(
        row_tuple_list, train, has_weight, has_fit_base_margin,
        has_predict_base_margin)
    X = _handle_missing_values(feature_matrix, missing)
    return X, y, w, b_m


def _process_row_tuple_iter(row_tuple_iter,
                            train,
                            has_weight,
                            missing,
                            has_validation,
                            has_fit_base_margin=False,
                            has_predict_base_margin=False):
    """
    If input is for train and has_validation=True, it will split the train data into train dataset
    and validation dataset, and return (train_X, train_y, train_w, train_b_m <-
    train base margin, val_X, val_y, val_w, val_b_m <- validation base margin)
    otherwise return (X, y, w, b_m <- base margin)
    """
    if train and has_validation:
        train_row_tuple_list, val_row_tuple_list = [], []
        for row_tuple in row_tuple_iter:
            if row_tuple.validationIndicator:
                val_row_tuple_list.append(row_tuple)
            else:
                train_row_tuple_list.append(row_tuple)
        train_X, train_y, train_w, train_b_m = _process_row_tuple_list(
            train_row_tuple_list, train, has_weight, missing,
            has_fit_base_margin, has_predict_base_margin)
        val_X, val_y, val_w, val_b_m = _process_row_tuple_list(
            val_row_tuple_list, train, has_weight, missing,
            has_fit_base_margin, has_predict_base_margin)
        return train_X, train_y, train_w, train_b_m, val_X, val_y, val_w, val_b_m
    else:
        return _process_row_tuple_list(list(row_tuple_iter), train, has_weight,
                                       missing, has_fit_base_margin,
                                       has_predict_base_margin)


def convert_partition_data_to_dmatrix(partition_data_iter,
                                      has_weight,
                                      xgb_model,
                                      has_validation,
                                      use_external_storage=False,
                                      file_prefix=None,
                                      external_storage_precision=5,
                                      dmatrix_kwargs=None):
    dmatrix_kwargs = dmatrix_kwargs or {}
    # if we are using external storage, we use a different approach for making the dmatrix
    if use_external_storage:
        if has_validation:
            train_file, validation_file = _stream_data_into_libsvm_file(
                partition_data_iter, has_weight, xgb_model.missing,
                has_validation, file_prefix, external_storage_precision)
            training_dmatrix = _create_dmatrix_from_file(
                train_file, "{}/train.cache".format(file_prefix),
                dmatrix_kwargs
            )
            val_dmatrix = _create_dmatrix_from_file(
                validation_file, "{}/val.cache".format(file_prefix),
                dmatrix_kwargs
            )
            return training_dmatrix, val_dmatrix
        else:
            train_file = _stream_data_into_libsvm_file(
                partition_data_iter, has_weight, xgb_model.missing,
                has_validation, file_prefix, external_storage_precision)
            training_dmatrix = _create_dmatrix_from_file(
                train_file, "{}/train.cache".format(file_prefix),
                dmatrix_kwargs
            )
            return training_dmatrix

    # if we are not using external storage, we use the standard method of parsing data.
    train_val_data = prepare_train_val_data(partition_data_iter, has_weight,
                                            xgb_model.missing, has_validation)
    if has_validation:
        train_X, train_y, train_w, _, val_X, val_y, val_w, _ = train_val_data
        training_dmatrix = DMatrix(data=train_X, label=train_y, weight=train_w, **dmatrix_kwargs)
        val_dmatrix = DMatrix(data=val_X, label=val_y, weight=val_w, **dmatrix_kwargs)
        return training_dmatrix, val_dmatrix
    else:
        train_X, train_y, train_w, _ = train_val_data
        training_dmatrix = DMatrix(data=train_X, label=train_y, weight=train_w, **dmatrix_kwargs)
        return training_dmatrix
