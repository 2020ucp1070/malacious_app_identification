import os
import pandas as pd


def fill_nan_with_mid_value(df):
    for column in df.columns:
        if df[column].dtype == 'float64':  # Check if the column has float64 dtype
            mid_value = df[column].mean()
            # mid_value = 0  # Calculate the median value
            # Fill NaN values with the mid value directly
            df[column] = df[column].fillna(mid_value)
    return df


def apply_limits(value, lower_limit, upper_limit):
    return min(max(value, lower_limit), upper_limit)


def update_features(input_folder, output_folder, feature_class_name, upper_limit=1e9, lower_limit=-1e9):
    # Get all CSV files in the input folder
    csv_files = [file for file in os.listdir(
        input_folder) if file.endswith('.csv')]

    for file in csv_files:
        # Define the file paths for input and output CSV files
        file_path = os.path.join(input_folder, file)
        output_file_path = os.path.join(output_folder, file)
        print(file_path)
        # Load the CSV file into a Pandas DataFrame
        df1 = pd.read_csv(file_path)

        # Calculate the required features from existing features
        df_new = pd.DataFrame({
            'duration': df1['flow_duration'],
            'total_fpackets': df1['tot_fwd_pkts'],
            'total_bpackets': df1['tot_bwd_pkts'],
            'total_fpktl': df1['totlen_fwd_pkts'],
            'total_bpktl': df1['totlen_bwd_pkts'],
            'min_fpktl': df1['fwd_pkt_len_min'],
            'min_bpktl': df1['bwd_pkt_len_min'],
            'max_fpktl': df1['fwd_pkt_len_max'],
            'max_bpktl': df1['bwd_pkt_len_max'],
            'mean_fpktl': df1['fwd_pkt_len_mean'],
            'mean_bpktl': df1['bwd_pkt_len_mean'],
            'std_fpktl': df1['fwd_pkt_len_std'],
            'std_bpktl': df1['bwd_pkt_len_std'],
            'total_fiat': df1['fwd_iat_tot'] + df1['bwd_iat_tot'],
            'total_biat': df1['fwd_iat_tot'] + df1['bwd_iat_tot'],
            'min_fiat': df1[['fwd_iat_min', 'bwd_iat_min']].min(axis=1),
            'min_biat': df1[['fwd_iat_min', 'bwd_iat_min']].min(axis=1),
            'max_fiat': df1[['fwd_iat_max', 'bwd_iat_max']].max(axis=1),
            'max_biat': df1[['fwd_iat_max', 'bwd_iat_max']].max(axis=1),
            'mean_fiat': (df1['fwd_iat_mean'] + df1['bwd_iat_mean']) / 2,
            'mean_biat': (df1['fwd_iat_mean'] + df1['bwd_iat_mean']) / 2,
            'std_fiat': ((df1['fwd_iat_std'] ** 2) + (df1['bwd_iat_std'] ** 2)) ** 0.5,
            'std_biat': ((df1['fwd_iat_std'] ** 2) + (df1['bwd_iat_std'] ** 2)) ** 0.5,
            'fpsh_cnt': df1['fwd_psh_flags'],
            'bpsh_cnt': df1['bwd_psh_flags'],
            'furg_cnt': df1['fwd_urg_flags'],
            'burg_cnt':  df1['bwd_urg_flags'],
            'total_fhlen': df1['fwd_header_len'],
            'total_bhlen': df1['bwd_header_len'],
            'fPktsPerSecond': df1['fwd_act_data_pkts'] / df1['flow_duration'],
            'bPktsPerSecond': df1['fwd_act_data_pkts'] / df1['flow_duration'],
            'flowPktsPerSecond': (df1['tot_fwd_pkts'] + df1['tot_bwd_pkts']) / df1['flow_duration'],
            'flowBytesPerSecond': (df1['totlen_fwd_pkts'] + df1['totlen_bwd_pkts']) / df1['flow_duration'],
            'min_flowpktl': df1[['fwd_pkt_len_min', 'bwd_pkt_len_min']].min(axis=1),
            'max_flowpktl': df1[['fwd_pkt_len_max', 'bwd_pkt_len_max']].max(axis=1),
            'mean_flowpktl': (df1['fwd_pkt_len_mean'] + df1['bwd_pkt_len_mean']) / 2,
            'std_flowpktl': ((df1['fwd_pkt_len_std'] ** 2) + (df1['bwd_pkt_len_std'] ** 2)) ** 0.5,
            'min_flowiat': df1[['fwd_iat_min', 'bwd_iat_min']].min(axis=1),
            'max_flowiat': df1[['fwd_iat_max', 'bwd_iat_max']].max(axis=1),
            'mean_flowiat': (df1['fwd_iat_mean'] + df1['bwd_iat_mean']) / 2,
            'std_flowiat': ((df1['fwd_iat_std'] ** 2) + (df1['bwd_iat_std'] ** 2)) ** 0.5,
            'flow_fin': df1['fin_flag_cnt'],
            'flow_syn': df1['syn_flag_cnt'],
            'flow_rst': df1['rst_flag_cnt'],
            'flow_psh': df1['psh_flag_cnt'],
            'flow_ack': df1['ack_flag_cnt'],
            'flow_urg': df1['urg_flag_cnt'],
            'flow_cwr': df1['cwe_flag_count'],
            'flow_ece': df1['ece_flag_cnt'],
            'downUpRatio': df1['down_up_ratio'],
            'avgPacketSize': df1['pkt_size_avg'],
            'fAvgSegmentSize': df1['fwd_seg_size_avg'],
            'fHeaderBytes': df1['fwd_byts_b_avg'],
            'fAvgBytesPerBulk': df1['fwd_blk_rate_avg'],
            'fAvgPacketsPerBulk': df1['fwd_pkts_b_avg'],
            'fAvgBulkRate': df1['fwd_blk_rate_avg'],
            'bVarianceDataBytes': df1['bwd_byts_b_avg'],
            'bAvgSegmentSize': df1['bwd_seg_size_avg'],
            'bAvgBytesPerBulk': df1['bwd_blk_rate_avg'],
            'bAvgPacketsPerBulk': df1['bwd_pkts_b_avg'],
            'bAvgBulkRate': df1['bwd_blk_rate_avg'],
            'sflow_fpacket': df1['subflow_fwd_pkts'],
            'sflow_fbytes': df1['subflow_fwd_byts'],
            'sflow_bpacket': df1['subflow_bwd_pkts'],
            'sflow_bbytes': df1['subflow_bwd_byts'],
            'min_active': df1['active_min'],
            'mean_active': df1['active_mean'],
            'max_active': df1['active_max'],
            'std_active': df1['active_std'],
            'min_idle': df1['idle_min'],
            'mean_idle': df1['idle_mean'],
            'max_idle': df1['idle_max'],
            'std_idle': df1['idle_std'],
            'FFNEPD': df1['totlen_fwd_pkts'] / (df1['fwd_pkts_b_avg'] + 1),
            'Init_Win_bytes_forward': df1['init_fwd_win_byts'],
            'Init_Win_bytes_backward': df1['init_fwd_win_byts'],
            'RRT_samples_clnt': df1['flow_iat_mean'] * df1['flow_pkts_s'],
            'Act_data_pkt_forward': df1['fwd_act_data_pkts'],
            'min_seg_size_forward': df1['fwd_seg_size_min'],
            'calss': feature_class_name
        })
        # Convert float64 columns to int64 in df_new
        float_to_int_columns = [
            'duration', 'min_fpktl', 'min_bpktl', 'max_fpktl', 'max_bpktl',
            'total_fiat', 'total_biat', 'min_fiat', 'min_biat', 'max_fiat', 'max_biat',
            'min_flowpktl', 'max_flowpktl', 'min_flowiat', 'max_flowiat',
            'fHeaderBytes', 'fAvgBytesPerBulk', 'fAvgPacketsPerBulk', 'fAvgBulkRate',
            'bAvgSegmentSize', 'bAvgBytesPerBulk', 'bAvgPacketsPerBulk', 'bAvgBulkRate',
            'min_active', 'max_active', 'min_idle', 'max_idle', 'FFNEPD', 'RRT_samples_clnt'
        ]

# Loop through the columns and convert them to int64
        for column in float_to_int_columns:
            df_new[column] = df_new[column].astype('int64')

        float_columns = df_new.select_dtypes(include=['float64']).columns
        # print(float_columns)
        df_new[float_columns] = df_new[float_columns].apply(
            lambda x: x.apply(lambda y: apply_limits(y, lower_limit, upper_limit)))

        df_new = fill_nan_with_mid_value(df_new)
        # Save the new DataFrame to the output file
        df_new.to_csv(output_file_path, index=False)


# Example usage
input_folder = './DATASET/BENIGN'
output_folder = './DATASET/BENIGN-1'
feature_class_name = 'benign'
update_features(input_folder, output_folder, feature_class_name)
# GeneralMalware
# benign
