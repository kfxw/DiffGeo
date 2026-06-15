from diffgeo.data import AirfoilDataset, load_airfoil_array, load_airfoil_names, read_airfoil_dat
from diffgeo.geometry import max_thickness_numpy

from .conftest import repo_root


def test_uiuc_split_and_dataset_load():
    root = repo_root()
    split = root / "data" / "splits" / "train_full_UIUC.txt"
    data_dir = root / "data" / "uiuc_airfoils" / "dat"
    names = load_airfoil_names(split, limit=3)
    assert len(names) == 3
    dataset = AirfoilDataset(data_dir=data_dir, split_file=split, num_points=64, limit=3)
    item = dataset[0]
    assert item["points"].shape == (64, 2)
    assert item["points"][:, 0].min() >= -1e-5
    assert item["points"][:, 0].max() <= 1.0 + 1e-5


def test_uiuc_selig_count_header_is_not_loaded_as_coordinate():
    root = repo_root()
    path = root / "data" / "uiuc_airfoils" / "dat" / "e582-il.dat"
    points = read_airfoil_dat(path)
    assert points[0, 0] < 0.01
    assert points[0, 1] < 0.01
    assert points[:, 0].max() <= 1.0 + 1e-5
    assert points[:, 1].max() < 0.2


def test_uiuc_airfoil_is_canonicalized_to_template_order():
    root = repo_root()
    data_dir = root / "data" / "uiuc_airfoils" / "dat"
    points = load_airfoil_array("e582-il", data_dir, 80)
    assert points.shape == (80, 2)
    assert points[0, 0] > 0.99
    assert points[len(points) // 2 - 1, 0] < 0.02
    assert points[-1, 0] > 0.99
    assert -0.08 < points[:, 1].min() < 0.0
    assert 0.08 < points[:, 1].max() < 0.2
    assert max_thickness_numpy(points) > 0.1
