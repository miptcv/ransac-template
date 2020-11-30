import sys, os.path, json, numpy as np


def generate_data(
        img_size: tuple, line_params: tuple,
        n_points: int, sigma: float, inlier_ratio: float
) -> np.ndarray:
    pass  # insert your code here


def compute_ransac_threshold(
        alpha: float, sigma: float
) -> float:
    pass  # insert your code here


def compute_ransac_iter_count(
        conv_prob: float, inlier_ratio: float
) -> int:
    pass  # insert your code here


def compute_line_ransac(
        data: np.ndarray, threshold: float, iter_count: int
) -> tuple:
    pass  # insert your code here


def detect_line(params: dict) -> tuple:
    data = generate_data(
        (params['w'], params['h']),
        (params['a'], params['b'], params['c']),
        params['n_points'], params['sigma'], params['inlier_ratio']
    )
    threshold = compute_ransac_threshold(
        params['alpha'], params['sigma']
    )
    iter_count = compute_ransac_iter_count(
        params['conv_prob'], params['inlier_ratio']
    )
    detected_line = compute_line_ransac(data, threshold, iter_count)
    return detected_line


def main():
    assert len(sys.argv) == 2
    params_path = sys.argv[1]
    assert os.path.exists(params_path)
    with open(params_path) as fin:
        params = json.load(fin)
    assert params is not None

    """
    params:
    line_params: (a,b,c) - line params (ax+by+c=0)
    img_size: (w, h) - size of the image
    n_points: count of points to be used

    sigma - Gaussian noise
    alpha - probability of point is an inlier

    inlier_ratio - ratio of inliers in the data
    conv_prob - probability of convergence
    """

    detected_line = detect_line(params)
    print(detected_line)


if __name__ == '__main__':
    main()
