import streamlit as st
import numpy as np
import pandas as pd
import re
import pickle

with open('gaussian.pkl', 'rb') as file:
    model = pickle.load(file)

with open('standardization.pkl', 'rb') as file:
    scaler = pickle.load(file)


def main():
    def map_ports_to_services(info, common_ports, existing_service):
            # If the existing service is not NaN, return it as is
            if not pd.isna(existing_service):
                return existing_service
            
            # Search for port numbers in the Info string
            ports = re.findall(r'\b\d+\b', info)
            # Check each port and return the corresponding service if found
            for port in ports:
                port = int(port)
                if port in common_ports:
                    return common_ports[port]
            return np.nan  # Return NaN if no matching port is found
    
    def map_service_to_protocol(service, existing_protocol, protocols_dict):
            # If the existing service is not NaN, return it as is
            if not pd.isna(existing_protocol):
                return existing_protocol
            
            # Check each port and return the corresponding service if found
            for key, protocol in protocols_dict.items():
                if service in protocol:
                    return key
            return np.nan  # Return NaN if no matching port is found
    
    # Function to assign value to src_bytes based on Source
    def assign_src_bytes(row, host_ip):
        if row['Source'] == host_ip:
            return row['Length']
        else:
            return 0
        
    # Function to assign value to src_bytes based on Source
    def assign_dst_bytes(row, host_ip):
        if row['Destination'] == host_ip:
            return row['Length']
        else:
            return 0
        
    # Define a function to preprocess the dataframe
    def preprocess_dataframe(df, scaler):
        # Define the transport protocols that will go into 'protocol_type'
        transport_protocols = ['TCP', 'UDP', 'ICMP']

        # Create 'protocol_type' and 'service' columns
        df['protocol_type'] = df['Protocol'].apply(lambda x: x if x in transport_protocols else np.nan)
        df['service'] = df['Protocol'].apply(lambda x: x if x not in transport_protocols else np.nan)

        common_ports = {
            80: "http",
            443: "https",
            21: "ftp",
            22: "ssh",
            25: "smtp",
            110: "pop3",
            143: "imap",
            53: "dns",
            67: "dhcp-server",
            68: "dhcp-client",
            3306: "mysql",
            5432: "postgresql",
            3389: "rdp",
            6660: "irc",
            6661: "irc",
            6662: "irc",
            6663: "irc",
            6664: "irc",
            6665: "irc",
            6666: "irc",
            6667: "irc",
            6668: "irc",
            6669: "irc"
        }
        # mapping services based on port information found in info
        df['service'] = df.apply(lambda row: map_ports_to_services(row['Info'], common_ports, row['service']), axis=1)

        # Apply lowercase transformation to all data except column names
        df.iloc[:, :] = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        protocols_dict = {
            'tcp': [
                'http', 'https', 'ftp', 'sftp', 'ssh', 'smtp', 'pop3', 'imap', 'mysql', 'postgresql', 'rdp',
                'irc', 'telnet', 'ldap', 'smb', 'sql', 'sip', 'imaps', 'pop3s', 'ftps', 'redis', 'memcached',
                'mongodb', 'mqtt', 'rtsp', 'dns', 'ntp', 'radius', 'kerberos', 'zookeeper', 'kafka',
                'cassandra', 'couchdb', 'elasticsearch', 'vnc', 'nfs', 'sqlsrv', 'webdav', 'xsmtp', 'smtps',
                'mysql', 'postgres', 'ssl', 'tlsv1.0', 'tlsv1.1', 'tlsv1.2', 'tlsv1.3', 'http2', 'http3'
            ],
            'udp': [
                'dns', 'dhcp', 'ntp', 'tftp', 'snmp', 'syslog', 'sdp', 'mDNS', 'rtp', 'rtcp', 'quic',
                'vnc', 'x11', 'radius', 'kerberos', 'nfs', 'sctp', 'smb', 'sdp', 'udp', 'dgram',
                'snmp', 'rtsps', 'sips', 'sctp', 'srp', 'srv', 'scp'
            ],
            'icmp': [
                'ping', 'traceroute', 'icmp echo request', 'icmp echo reply', 'icmp destination unreachable',
                'icmp time exceeded', 'icmp parameter problem', 'icmp source quench', 'icmp redirect',
                'icmp timestamp request', 'icmp timestamp reply', 'icmp address mask request', 'icmp address mask reply'
            ]
        }
        # Using commonly found services provided by differnt protocol to map protocols based on services
        df['protocol_type'] = df.apply(lambda row: map_service_to_protocol(row['service'], row['protocol_type'], protocols_dict), axis=1)

        # Get the most frequent value (mode) from the 'Source' column as host ip is mostly interacting so to identify host_Ip
        most_frequent_value = df['Source'].mode()[0]


        # Apply the function to create a new column src_bytes
        df['src_bytes'] = df.apply(lambda row: assign_src_bytes(row, host_ip=most_frequent_value), axis=1)

        # Apply the function to create a new column src_bytes
        df['dst_bytes'] = df.apply(lambda row: assign_dst_bytes(row, host_ip=most_frequent_value), axis=1)


        df['Timestamp'] = pd.to_datetime(df['Time'])

        # Iterate over each row to compute count and srv_count
        for i, row in df.iterrows():
            # Get the current timestamp and filters
            current_time = row['Timestamp']
            current_destination = row['Destination']
            current_service = row['service']
            
            # Filter rows within the past two seconds
            time_window = (df['Timestamp'] >= current_time - pd.Timedelta(seconds=2)) & (df['Timestamp'] <= current_time)
            
            # Compute count for connections to the same destination
            count = max(df[time_window & (df['Destination'] == current_destination)].shape[0] - 1, 0)
            df.at[i, 'count'] = count

            # Compute srv_count for connections to the same service
            srv_count = max(df[time_window & (df['service'] == current_service)].shape[0] - 1, 0)
            df.at[i, 'srv_count'] = srv_count

        # features selection as per the training details
        selected_features = ['protocol_type', 'service', 'src_bytes', 'count', 'srv_count', 'dst_bytes']

        df_with_selected_features = df[selected_features]

        # List of categorical columns to be decoded
        categorical_columns = [col for col in df_with_selected_features.columns if df_with_selected_features[col].dtype == 'object']

        # Loop through each categorical column, load the encoder, and transform the test/new data
        for col in categorical_columns:
            # Load the saved encoder from the pickle file
            encoder_filename = f'{col}_encoder.pkl'
            
            with open(encoder_filename, 'rb') as file:
                encoder = pickle.load(file)

            # Transform the test data (or any new data) using the loaded encoder
            df_with_selected_features[col] = encoder.transform(df_with_selected_features[[col]])+1

        numeric_columns = ['src_bytes', 'count', 'srv_count']

        scaled_data = scaler.fit_transform(df_with_selected_features[numeric_columns])
        # Convert the scaled data to a DataFrame with the same column names
        df_scaled = pd.DataFrame(scaled_data, columns=numeric_columns)

        # Replace the original columns with the scaled data
        df_with_selected_features[numeric_columns] = df_scaled
        return df_with_selected_features

    # Streamlit app layout
    st.title('Network Anamoly detection')
    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")


    # Button to trigger analysis
    if st.button('Analyze'):
        if uploaded_file is not None:
            # Read the CSV file
            input_df = pd.read_csv(uploaded_file)
            
            # preprocess the dataframe
            processed_df = preprocess_dataframe(input_df, scaler)

            # sorting dataframe in alphabetic order
            sorted_final_df = processed_df.sort_index(axis=1)
            # making model prediction
            input_df['cluster_gaussian'] = model.predict(sorted_final_df)

            numbers = input_df.groupby('cluster_gaussian').size()

            st.write("### Data Analysis")
            st.write("### Cluster 0 represents normal and Cluster 1 represents anamoly.")
            st.write(numbers)

            # getting detail of anamolies like source, destination, and info
            # Filter and select columns
            if 'cluster_gaussian' in input_df.columns:
                anomaly_details = input_df[input_df['cluster_gaussian'] == 1][['source', 'destination', 'info']]
                
                if not anomaly_details.empty:
                    st.write("### Anomaly Details")
                    st.write(anomaly_details)

                    # Prepare the CSV file for download
                    @st.cache
                    def convert_df_to_csv(df):
                        return input_df.to_csv(index=False)

                    csv_data = convert_df_to_csv(anomaly_details)

                    st.download_button(
                        label="Download Anomaly Details CSV",
                        data=csv_data,
                        file_name='anomaly_details.csv',
                        mime='text/csv'
                    )
                else:
                    st.write("No anomalies found.")
            else:
                st.warning("The required columns are not present in the CSV file.")
        else:
            st.warning("Please upload a CSV file.")

if __name__ == '__main__':
    main()
