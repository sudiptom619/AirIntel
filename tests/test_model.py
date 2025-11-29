from models.predict import predict_point


def _check_res(res):
    # basic shape checks
    assert isinstance(res, dict)
    assert "score" in res
    assert 0 <= res["score"] <= 100
    assert "components" in res
    # components should be a dict with expected keys
    comps = res["components"]
    assert isinstance(comps, dict)
    expected = {"pollution_score", "traffic_score", "industry_score", "green_score", "pop_score", "flood_score", "elev_score"}
    # at least some expected keys present
    assert len(set(comps.keys()).intersection(expected)) >= 3


def test_predict_point_basic():
    # central Kolkata-ish point
    res = predict_point(22.5726, 88.3639)
    _check_res(res)


def test_predict_point_other_locations():
    # a few other sample points (nearby grid cells)
    pts = [
        (22.542444, 88.334489),
        (22.550741, 88.35245),
        (22.561, 88.37),
    ]
    for lat, lon in pts:
        r = predict_point(lat, lon)
        _check_res(r)
