"""
generate_dashboard.py
Simple wrapper to generate dashboard from logs
"""

from dashboard_generator import generate_dashboard
import sys

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  🐕 LLM WATCHDOG - Dashboard Generator")
    print("="*70 + "\n")
    
    # Check for custom log file
    log_file = "qa_monitoring_logs.jsonl"
    output_file = "dashboard.html"
    
    if "--log" in sys.argv:
        idx = sys.argv.index("--log")
        if idx + 1 < len(sys.argv):
            log_file = sys.argv[idx + 1]
    
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]
    
    success = generate_dashboard(log_file, output_file)
    
    if success:
        print(f"\n✅ Success! Open {output_file} in your browser")
    else:
        print("\n❌ Failed to generate dashboard")