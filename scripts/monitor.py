import subprocess
import psutil
import time

class TrafficMonitor:
    def __init__(self, src_host, dst_host):
        self.src_host = src_host
        self.dst_host = dst_host
    
    def measure_traffic_alternative(self, duration=10):
        """
        Alternative traffic measurement method using multiple approaches
        """
        try:
            # Method 1: Use iftop-like parsing
            return self._measure_with_iftop(duration)
        except Exception:
            try:
                # Method 2: Use psutil network counters
                return self._measure_with_psutil(duration)
            except Exception:
                try:
                    # Method 3: Use simple netstat parsing
                    return self._measure_with_netstat(duration)
                except Exception as e:
                    print(f"All traffic measurement methods failed: {e}")
                    return None
    
    def _measure_with_iftop(self, duration):
        """
        Measure traffic using iftop-like approach
        """
        try:
            # # Ensure iftop is installed
            # subprocess.run(['sudo', 'apt-get', 'install', '-y', 'iftop'], check=True)
            
            # Run iftop with specific parameters
            cmd = [
                'sudo', 'iftop', 
                '-t',  # Text mode
                '-s', str(duration),  # Sample duration
                '-L', '1',  # Limit to 1 line of output
                '-i', self.src_host.defaultIntf().name  # Source interface
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=duration + 5
            )
            
            # Parse iftop output (you may need to adjust parsing)
            if result.stdout:
                # Example parsing (adjust based on actual iftop output)
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'Total' in line:
                        # Extract numeric values
                        parts = line.split()
                        total_traffic = float(parts[1])
                        return total_traffic
            
            return None
        
        except Exception as e:
            print(f"iftop measurement failed: {e}")
            return None
    
    def _measure_with_psutil(self, duration):
        """
        Measure traffic using psutil network counters
        """
        try:
            # Get initial network counters
            initial_counters = psutil.net_io_counters()
            
            # Wait for specified duration
            time.sleep(duration)
            
            # Get final network counters
            final_counters = psutil.net_io_counters()
            
            # Calculate traffic
            bytes_sent = final_counters.bytes_sent - initial_counters.bytes_sent
            bytes_recv = final_counters.bytes_recv - initial_counters.bytes_recv
            
            # Convert to Mbps
            total_traffic_mbps = ((bytes_sent + bytes_recv) * 8) / (duration * 1024 * 1024)
            
            return total_traffic_mbps
        
        except Exception as e:
            print(f"psutil measurement failed: {e}")
            return None
    
    def _measure_with_netstat(self, duration):
        """
        Measure traffic using netstat
        """
        try:
            # Run netstat to get network statistics
            initial_cmd = ['netstat', '-i']
            final_cmd = ['netstat', '-i']
            
            # Get initial stats
            initial_output = subprocess.check_output(initial_cmd, text=True)
            
            # Wait for duration
            time.sleep(duration)
            
            # Get final stats
            final_output = subprocess.check_output(final_cmd, text=True)
            
            # Parse and calculate traffic (simplified)
            # You'll need to implement more robust parsing
            return self._parse_netstat_output(initial_output, final_output)
        
        except Exception as e:
            print(f"netstat measurement failed: {e}")
            return None
    
    def _parse_netstat_output(self, initial_output, final_output):
        """
        Parse netstat output to extract traffic
        Implement sophisticated parsing logic
        """
        # Placeholder implementation
        return 10.0  # Example traffic value
    
    def fallback_traffic_estimation(self, predicted_traffic):
        """
        Fallback method if all measurement techniques fail
        """
        print("Using predicted traffic as fallback")
        return predicted_traffic