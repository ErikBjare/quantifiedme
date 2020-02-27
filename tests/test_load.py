def test_load_qslang():
    from quantifiedme.qslang import load_df
    assert not load_df().empty


def test_load_activitywatch():
    from quantifiedme.activitywatch import load_category_df
    assert not load_category_df().empty


def test_load_location():
    from quantifiedme.location import load_all_dfs
    assert load_all_dfs()


def test_load_habitbull():
    from quantifiedme.habitbull import load_df
    assert not load_df().empty


def test_load_oura():
    from quantifiedme.oura import load_sleep_df, load_activity_df

    assert not load_sleep_df().empty

    activity_df = load_activity_df()
    assert not activity_df.empty
