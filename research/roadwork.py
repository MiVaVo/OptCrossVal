def windowed_diversify_portfilio(df, window_size=10, error_on_ws=False):
    if window_size <= df.shape[1] * 3:
        message = f"Window size must be greater than {df.shape[1] * 3}, but got {window_size}"
        if error_on_ws:
            raise ValueError(message)
        else:
            Warning(f"Increasing window size to {df.shape[1] * 3}")
            window_size = df.shape[1] * 3
    list_df = [df[i:i + window_size] for i in range(0, df.shape[0], window_size)]
    res_dict = {"start": [],
                "end": [],
                "weights": [],
                "stds": []}
    n_iters = 10
    # for df in list_df:
    #     weights_accumulated = boostraped_diversify_portfolio(df, n_iters=n_iters)
    #     res_dict['start'].append(sorted(df.index)[0])
    #     res_dict['end'].append(sorted(df.index)[-1])
    #     res_dict['weights'].append(weights_accumulated)

    # for info in np.asarray(res_dict['weights']).T:
    #     print(info)
    # z,p=get_z_score_for_diff_in_mean(info[0][0],info[0][1],info[1][0],info[1][1],n1=n_iters,n2=n_iters)
    # print(z,1-p)


windowed_diversify_portfilio(df_day.dropna())
