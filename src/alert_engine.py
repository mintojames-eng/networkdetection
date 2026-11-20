"""
Real-time Alerting & Incident Response Engine
Generates structured security alerts (JSON/Syslog) and automated mitigation commands.
"""

from datetime import datetime
import json

SEVERITY_LEVELS = {
    "DDoS": "CRITICAL",
    "DoS": "HIGH",
    "BruteForce": "HIGH",
    "Probe": "MEDIUM",
    "Unknown": "LOW"
}

MITIGATION_TEMPLATES = {
    "DDoS": "iptables -I INPUT 1 -s {src_ip} -j DROP && ip route add blackhole {src_ip}",
    "DoS": "iptables -A INPUT -s {src_ip} -p tcp --syn -m connlimit --connlimit-above 50 -j REJECT",
    "BruteForce": "fail2ban-client set sshd banip {src_ip} && iptables -A INPUT -s {src_ip} -p tcp --dport 22 -j DROP",
    "Probe": "iptables -A INPUT -s {src_ip} -m recent --name portscan --set -j DROP",
    "Unknown": "iptables -A INPUT -s {src_ip} -j LOG --log-prefix '[NIDS-SUSPICIOUS] '"
}

class AlertEngine:
    def __init__(self, log_to_file=False, log_path="alerts.log"):
        self.log_to_file = log_to_file
        self.log_path = log_path

    def generate_normal_log(self, traffic_id, latency_ms, telemetry):
        """Generates standard benign audit event."""
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "traffic_id": traffic_id,
            "status": "NORMAL",
            "action": "ALLOWED",
            "telemetry": telemetry,
            "inspection_latency_ms": round(latency_ms, 3)
        }
        return event

    def generate_attack_alert(self, traffic_id, attack_type, confidence_pct, latency_ms, telemetry):
        """Generates rich incident response alert for malicious flows."""
        severity = SEVERITY_LEVELS.get(attack_type, "MEDIUM")
        src_ip = telemetry.get("src_ip", "0.0.0.0")
        mitigation_cmd = MITIGATION_TEMPLATES.get(attack_type, MITIGATION_TEMPLATES["Unknown"]).format(src_ip=src_ip)
        
        alert = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "traffic_id": traffic_id,
            "status": "ATTACK_DETECTED",
            "level_1_filter": "MALICIOUS",
            "level_2_attack_type": attack_type,
            "confidence": f"{confidence_pct:.2f}%",
            "severity": severity,
            "telemetry": telemetry,
            "automated_mitigation": {
                "rule_name": f"BLOCK_{attack_type.upper()}_{src_ip}",
                "firewall_command": mitigation_cmd
            },
            "inspection_latency_ms": round(latency_ms, 3)
        }
        
        if self.log_to_file:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert) + "\n")
                
        return alert
