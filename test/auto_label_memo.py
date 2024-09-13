# # 오류(SettingWithCopyError 발생)
# pd.set_option('mode.chained_assignment', 'raise') # SettingWithCopyError

# # 경고(SettingWithCopyWarning 발생, 기본 값입니다)
# pd.set_option('mode.chained_assignment', 'warn') # SettingWithCopyWarning

# 무시

# def dr_outlier(df): #IQR method
#     quartile_1 = df.quantile(0.25)
#     quartile_3 = df.quantile(0.75)
#     IQR = quartile_3 - quartile_1
#     condition = (df < (quartile_1 - 1.5 * IQR)) | (df > (quartile_3 + 1.5 * IQR))
#     condition = condition.any(axis=1)
#     search_df = df[condition]

#     return search_df, df.drop(search_df.index, axis=0)

        # a = df2drop['Vy'].mean()/df2drop['Vx'].mean()
        # b = df1drop['Y'].mean() - a * df1drop['X'].mean()
        
        # if a < 0:
        #     PF_vrf = df1drop['Y'] < a * df1drop['X'] + b
        # else:
        #     PF_vrf = df1drop['Y'] > a * df1drop['X'] + b