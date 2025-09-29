#!/usr/bin/env python3
"""
Sample Files Generator for LLM Training and Testing

This script creates realistic sample files for testing the PDF, CSV, and text processors.
Generates customer support tickets, chat logs, and documentation samples.
"""

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any


class SampleDataGenerator:
    """Generate realistic sample data for various file formats."""
    
    def __init__(self):
        # Sample data pools for generating realistic content
        self.customer_names = [
            "Emma Johnson", "Liam Smith", "Olivia Brown", "Noah Davis", "Ava Wilson",
            "William Garcia", "Isabella Martinez", "James Anderson", "Sophia Taylor", "Benjamin Thomas",
            "Charlotte Jackson", "Lucas White", "Mia Harris", "Henry Martin", "Amelia Thompson",
            "Alexander Garcia", "Harper Clark", "Michael Rodriguez", "Evelyn Lewis", "Daniel Lee"
        ]
        
        self.companies = [
            "TechCorp Solutions", "Global Dynamics", "Innovative Systems", "DataFlow Inc",
            "CloudTech Partners", "Digital Innovations", "Smart Solutions LLC", "Future Systems",
            "NextGen Technologies", "Quantum Enterprises", "Alpha Industries", "Beta Solutions",
            "Gamma Corporation", "Delta Dynamics", "Epsilon Systems"
        ]
        
        self.products = [
            "CloudSync Pro", "DataManager", "SecureVault", "WorkflowMax", "AnalyticsPro",
            "MobileConnect", "ServerGuard", "BackupMaster", "NetworkOptimizer", "TeamCollab"
        ]
        
        self.issues = [
            "Login authentication failure",
            "Unable to sync data across devices",
            "Payment processing error",
            "Mobile app crashes on startup",
            "File upload timeout",
            "Password reset not working",
            "Slow performance during peak hours",
            "Integration API returning errors",
            "Email notifications not received",
            "Dashboard loading blank page",
            "Export function corrupting files",
            "User permissions not updating",
            "Two-factor authentication bypass",
            "Database connection timeouts",
            "License activation issues"
        ]
        
        self.categories = [
            "Technical Issue", "Account Management", "Billing", "Feature Request",
            "Bug Report", "Integration", "Performance", "Security", "Training", "General Inquiry"
        ]
        
        self.priorities = ["Low", "Medium", "High", "Critical"]
        self.statuses = ["Open", "In Progress", "Pending Customer", "Resolved", "Closed"]
        
        self.agents = [
            "Sarah Mitchell", "David Chen", "Maria Rodriguez", "Alex Thompson",
            "Jennifer Wu", "Michael Brown", "Lisa Park", "Robert Kim"
        ]
        
        # Sample responses and solutions
        self.responses = [
            "Thank you for contacting support. I've reviewed your issue and will investigate immediately.",
            "I can see the problem you're experiencing. Let me walk you through the solution step by step.",
            "This appears to be a known issue that we're actively working to resolve. Here's a temporary workaround:",
            "I've escalated this to our engineering team for a permanent fix. You should see resolution within 24-48 hours.",
            "Based on your description, this seems to be a configuration issue. Let me help you adjust the settings.",
            "I've reset your account permissions. Please try logging in again and let me know if you continue to experience issues.",
            "This is related to our recent system update. I've applied a hotfix to your account that should resolve the problem.",
            "I can reproduce the issue on our end. I've created a bug report and our development team will prioritize this fix."
        ]
    
    def generate_customer_support_csv(self, filename: str = "customer_support_tickets.csv", num_tickets: int = 150) -> str:
        """Generate realistic customer support tickets CSV."""
        
        tickets = []
        
        for i in range(1, num_tickets + 1):
            # Generate realistic timestamp within last 3 months
            days_ago = random.randint(0, 90)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            
            timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
            
            # Create realistic ticket
            customer = random.choice(self.customer_names)
            company = random.choice(self.companies)
            product = random.choice(self.products)
            issue = random.choice(self.issues)
            category = random.choice(self.categories)
            priority = random.choice(self.priorities)
            status = random.choice(self.statuses)
            agent = random.choice(self.agents) if status != "Open" else ""
            
            # Generate realistic descriptions
            descriptions = [
                f"User experiencing {issue.lower()} when using {product}. Issue started {days_ago} days ago.",
                f"Customer from {company} reports {issue.lower()}. Affects multiple users in their organization.",
                f"Critical issue with {product} - {issue.lower()}. Customer from {company} needs immediate assistance.",
                f"Recurring problem reported by multiple users at {company}. {issue} affecting productivity.",
                f"Customer experiencing {issue.lower()} for the past {random.randint(1, 7)} days. Escalation requested.",
                f"New user from {company} having trouble with {product}. {issue} preventing account setup.",
                f"Premium customer reporting {issue.lower()}. Service level agreement requires 2-hour response.",
                f"Mobile app issue: {issue} on both iOS and Android devices. Affects user login process.",
                f"Integration problem with third-party system. {issue} causing data sync failures.",
                f"Billing dispute related to {issue.lower()}. Customer requesting immediate refund processing."
            ]
            
            description = random.choice(descriptions)
            
            # Generate resolution notes based on status
            resolution = ""
            if status == "Resolved":
                resolutions = [
                    "Issue resolved by resetting user credentials and clearing cache.",
                    "Problem fixed with latest software update deployed to customer account.",
                    "Configuration issue resolved. Settings updated per customer requirements.",
                    "Bug fix applied. Customer confirmed issue no longer occurring.",
                    "Account permissions updated. Customer can now access all required features.",
                    "Server maintenance completed. Performance issues resolved.",
                    "Integration endpoint updated. Data sync working properly now.",
                    "Refund processed successfully. Customer satisfied with resolution."
                ]
                resolution = random.choice(resolutions)
            elif status == "In Progress":
                resolutions = [
                    "Engineering team investigating root cause. Updates to follow.",
                    "Working with customer to gather additional diagnostic information.",
                    "Coordinating with product team for permanent fix implementation.",
                    "Temporary workaround provided while final solution is developed."
                ]
                resolution = random.choice(resolutions)
            
            # Calculate resolution time for closed tickets
            resolution_hours = ""
            if status in ["Resolved", "Closed"]:
                hours = random.randint(1, 72)
                resolution_hours = f"{hours}h"
            
            ticket = {
                'ticket_id': f'TKT-{i:05d}',
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'customer_name': customer,
                'customer_email': f"{customer.lower().replace(' ', '.')}@{company.lower().replace(' ', '').replace(',', '')}.com",
                'company': company,
                'product': product,
                'category': category,
                'priority': priority,
                'status': status,
                'subject': issue,
                'description': description,
                'assigned_agent': agent,
                'resolution_notes': resolution,
                'resolution_time': resolution_hours,
                'customer_satisfaction': random.choice([3, 4, 5]) if status == "Resolved" else "",
                'tags': f"{category.lower()},{priority.lower()},{product.lower().replace(' ', '')}"
            }
            
            tickets.append(ticket)
        
        # Write to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = tickets[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tickets)
        
        print(f"Created {filename} with {num_tickets} customer support tickets")
        return filename
    
    def generate_chat_logs_txt(self, filename: str = "internal_support_chat.txt", num_days: int = 7) -> str:
        """Generate realistic internal support team chat logs."""
        
        support_team = [
            "Sarah_Mitchell", "David_Chen", "Maria_Rodriguez", "Alex_Thompson", 
            "Jennifer_Wu", "Michael_Brown", "Lisa_Park", "Robert_Kim"
        ]
        
        chat_topics = [
            "escalation", "system_outage", "customer_complaint", "feature_request",
            "bug_report", "training", "policy_update", "team_meeting"
        ]
        
        # Generate chat logs
        chat_content = []
        chat_content.append("=" * 60)
        chat_content.append("YNC E-COMMERCE PLATFORM - INTERNAL SUPPORT CHAT LOGS")
        chat_content.append(f"Generated for Assignment: Customer Support Triage Agent")
        chat_content.append(f"Date Range: {datetime.now() - timedelta(days=num_days)} to {datetime.now()}")
        chat_content.append("=" * 60)
        chat_content.append("")
        
        for day in range(num_days):
            current_date = datetime.now() - timedelta(days=num_days - day - 1)
            chat_content.append(f"--- {current_date.strftime('%Y-%m-%d')} ---")
            
            # Generate 5-15 messages per day
            daily_messages = random.randint(8, 20)
            
            for _ in range(daily_messages):
                hour = random.randint(9, 18)
                minute = random.randint(0, 59)
                timestamp = current_date.replace(hour=hour, minute=minute)
                
                agent = random.choice(support_team)
                topic = random.choice(chat_topics)
                
                messages = {
                    "escalation": [
                        f"Need to escalate TKT-{random.randint(10000, 99999)} - customer threatening legal action",
                        f"Premium customer experiencing critical issue with payment processing",
                        f"Multiple complaints about same bug - should we create incident?",
                        f"Customer satisfaction score dropped to 2/5, need immediate attention"
                    ],
                    "system_outage": [
                        f"API endpoints showing 500 errors - investigating now",
                        f"Database performance degraded, query times > 10 seconds",
                        f"CDN issues affecting file uploads in EU region",
                        f"Third-party payment gateway down, customers can't complete purchases"
                    ],
                    "customer_complaint": [
                        f"Customer from TechCorp very upset about data loss incident",
                        f"Received complaint about rude support agent - need to review call",
                        f"Customer threatening to cancel due to repeated login issues",
                        f"Social media complaint gaining traction - PR team notified"
                    ],
                    "feature_request": [
                        f"Multiple requests for bulk export functionality this week",
                        f"Enterprise customers asking for SSO integration",
                        f"Mobile app users want offline mode capability",
                        f"Customers requesting better filtering options in dashboard"
                    ],
                    "bug_report": [
                        f"Confirmed bug in mobile app - crashes when uploading large files",
                        f"Email notifications have wrong timestamps in certain timezones",
                        f"Search function returns duplicate results for specific queries",
                        f"Export feature corrupting CSV files with special characters"
                    ],
                    "training": [
                        f"New agent training session scheduled for next week",
                        f"Updated knowledge base articles for common issues",
                        f"Policy changes effective immediately - please review",
                        f"Customer communication templates updated in system"
                    ],
                    "policy_update": [
                        f"New refund policy allows 60-day returns instead of 30",
                        f"Updated SLA: Critical tickets must respond within 1 hour",
                        f"Changed escalation procedure - now requires manager approval",
                        f"New data retention policy affects customer records"
                    ],
                    "team_meeting": [
                        f"Daily standup in 10 minutes - conference room B",
                        f"Sprint retrospective showed 15% improvement in resolution time",
                        f"Q3 customer satisfaction scores improved to 4.2/5 average",
                        f"Next week focus: reducing average response time to under 2 hours"
                    ]
                }
                
                message_text = random.choice(messages[topic])
                
                chat_content.append(f"[{timestamp.strftime('%H:%M')}] {agent}: {message_text}")
                
                # Occasionally add follow-up responses
                if random.random() < 0.3:
                    responder = random.choice([a for a in support_team if a != agent])
                    follow_ups = [
                        "Got it, I'll handle this immediately",
                        "Thanks for the heads up, investigating now",
                        "Added to my queue, will update in 30 minutes",
                        "Escalating to engineering team",
                        "Customer has been notified, waiting for response",
                        "Fixed - deployed hotfix to production"
                    ]
                    follow_up = random.choice(follow_ups)
                    follow_time = timestamp + timedelta(minutes=random.randint(2, 30))
                    chat_content.append(f"[{follow_time.strftime('%H:%M')}] {responder}: {follow_up}")
            
            chat_content.append("")
        
        # Add some emergency escalations and critical incidents
        chat_content.append("--- CRITICAL INCIDENTS LOG ---")
        chat_content.append("")
        
        incidents = [
            {
                "time": "2024-09-20 14:32",
                "type": "SYSTEM OUTAGE",
                "details": "Payment processing system down - affecting all transactions. Engineering team investigating."
            },
            {
                "time": "2024-09-18 09:15", 
                "type": "SECURITY ALERT",
                "details": "Unusual login activity detected from IP range 185.220.xxx.xxx. Security team notified."
            },
            {
                "time": "2024-09-15 16:45",
                "type": "DATA BREACH",
                "details": "Potential data exposure in export function. Immediately disabled feature. Legal team informed."
            }
        ]
        
        for incident in incidents:
            chat_content.append(f"[{incident['time']}] ALERT: {incident['type']}")
            chat_content.append(f"Details: {incident['details']}")
            chat_content.append("")
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(chat_content))
        
        print(f"Created {filename} with {num_days} days of chat logs")
        return filename
    
    def generate_support_policy_pdf_content(self) -> str:
        """Generate comprehensive support policy document content."""
        
        content = """YNC E-COMMERCE PLATFORM
CUSTOMER SUPPORT POLICIES AND PROCEDURES
Version 3.2 | Effective Date: September 2024

TABLE OF CONTENTS
1. Introduction and Overview
2. Support Ticket Classification
3. Priority Levels and SLA Requirements  
4. Escalation Procedures
5. Refund and Return Policies
6. Communication Guidelines
7. Quality Assurance Standards
8. Emergency Response Procedures

1. INTRODUCTION AND OVERVIEW

Purpose
This document outlines the comprehensive customer support policies and procedures for YNC E-Commerce Platform. All support agents must familiarize themselves with these guidelines to ensure consistent, high-quality customer service.

Scope
These policies apply to all customer-facing support interactions including:
- Email support tickets
- Live chat sessions  
- Phone support calls
- Social media inquiries
- Community forum responses

2. SUPPORT TICKET CLASSIFICATION

Primary Categories
TECHNICAL ISSUES
- Login and authentication problems
- Software bugs and glitches
- Integration and API issues
- Performance and connectivity problems
- Mobile application issues

ACCOUNT MANAGEMENT
- Profile and settings modifications
- Password resets and security
- Subscription changes and upgrades
- User permission management
- Account deactivation requests

BILLING AND PAYMENTS
- Payment processing errors
- Invoice and billing inquiries
- Refund and credit requests
- Subscription billing issues
- Tax and compliance questions

PRODUCT SUPPORT
- Feature explanations and tutorials
- Best practice recommendations
- Configuration assistance
- Data import/export help
- Third-party integrations

3. PRIORITY LEVELS AND SLA REQUIREMENTS

CRITICAL (P1)
- System-wide outages affecting all users
- Security breaches or data exposure
- Payment processing completely down
- Data loss or corruption

Response Time: 15 minutes
Resolution Time: 2 hours maximum
Escalation: Immediate to engineering

HIGH (P2)  
- Individual customer system down
- Premium customer issues
- Financial transaction errors
- API endpoints failing

Response Time: 1 hour
Resolution Time: 4 hours
Escalation: After 2 hours if unresolved

MEDIUM (P3)
- Feature not working as expected
- Performance issues affecting productivity
- Integration problems with workarounds
- Standard billing inquiries  

Response Time: 4 hours
Resolution Time: 24 hours
Escalation: After 12 hours if complex

LOW (P4)
- General questions and tutorials
- Feature requests and feedback
- Minor cosmetic issues
- Documentation requests

Response Time: 24 hours  
Resolution Time: 72 hours
Escalation: Rarely required

4. ESCALATION PROCEDURES

Level 1: Support Agent
- Initial customer contact and basic troubleshooting
- Access to knowledge base and standard solutions
- Authority to process refunds up to $500
- Can provide account credits up to $100

Level 2: Senior Agent/Team Lead
- Complex technical issues requiring advanced knowledge
- Customer satisfaction issues and complaints
- Refund authorization from $500-$2000
- Policy exception approvals

Level 3: Support Manager
- High-value customer issues
- Legal or compliance matters
- Major refunds over $2000
- Process improvements and policy changes

Level 4: Engineering/Product Teams
- Software bugs requiring code changes
- Infrastructure and system issues
- Feature development and enhancements  
- Security vulnerability responses

5. REFUND AND RETURN POLICIES

Standard Refund Policy
- Full refunds available within 60 days of purchase
- Prorated refunds for subscription cancellations
- No refunds for services already consumed
- Processing time: 5-7 business days

Exceptional Circumstances
- Service outages exceeding 4 hours: Automatic credit
- Data loss due to platform error: Full refund + damages
- Security breach affecting customer: Case-by-case basis
- Billing errors: Immediate correction and credit

Refund Authorization Levels
- Support Agent: Up to $500
- Senior Agent: $500 - $2,000  
- Manager: $2,000 - $10,000
- Director: Above $10,000

6. COMMUNICATION GUIDELINES

Professional Standards
- Always use customer's name in communications
- Respond with empathy and understanding
- Provide clear, step-by-step instructions
- Confirm understanding before closing tickets
- Follow up within 24 hours of resolution

Language and Tone
- Professional but friendly tone
- Avoid technical jargon unless necessary
- Provide context for any delays
- Always thank customers for their patience
- End with clear next steps or resolution

Response Templates
ACKNOWLEDGMENT:
"Thank you for contacting YNC Support. I understand you're experiencing [issue description]. I'm here to help resolve this for you promptly."

RESOLUTION:
"I'm pleased to confirm that your issue has been resolved. Here's a summary of the solution implemented: [details]. Please let me know if you need any additional assistance."

ESCALATION:
"I want to ensure you receive the best possible support for this complex issue. I'm escalating your case to our specialist team who will contact you within [timeframe]."

7. QUALITY ASSURANCE STANDARDS

Response Quality Metrics
- First Contact Resolution Rate: Target 70%
- Customer Satisfaction Score: Target 4.0/5.0
- Average Response Time: Target 2 hours
- Average Resolution Time: Target 12 hours

Monitoring and Review
- Random ticket audits: 10% of all tickets
- Customer feedback surveys: All resolved tickets
- Agent performance reviews: Monthly
- Policy compliance checks: Weekly

8. EMERGENCY RESPONSE PROCEDURES

System Outages
1. Immediate notification to engineering team
2. Status page update within 15 minutes  
3. Customer communication via email/SMS
4. Hourly updates until resolution
5. Post-incident report within 24 hours

Security Incidents
1. Isolate affected systems immediately
2. Notify security team and management
3. Document all actions taken
4. Customer notification if data affected
5. Coordinate with legal/PR teams

Data Loss Events
1. Stop all related processes immediately
2. Assess scope and impact
3. Notify affected customers within 2 hours
4. Implement recovery procedures
5. Provide regular status updates

APPENDICES

Appendix A: Contact Information
- Engineering Team: engineering@ync.com
- Security Team: security@ync.com  
- Management: support-mgmt@ync.com
- Legal Team: legal@ync.com

Appendix B: System Status Pages
- Main Platform: status.ync.com
- API Services: api-status.ync.com
- Payment Systems: payments-status.ync.com

Appendix C: Knowledge Base Links
- Common Issues: kb.ync.com/common
- API Documentation: docs.ync.com/api
- Video Tutorials: help.ync.com/videos

This document is reviewed quarterly and updated as needed. All support team members must acknowledge reading and understanding these policies.

Document Owner: Director of Customer Success
Last Updated: September 15, 2024
Next Review Date: December 15, 2024"""

        return content


def create_all_sample_files():
    """Create all sample files for the assignment."""
    print("Creating Sample Files for YNC E-Commerce Support Triage Agent Assignment")
    print("=" * 70)
    
    generator = SampleDataGenerator()
    
    # Create customer support tickets CSV
    csv_file = generator.generate_customer_support_csv("ync_customer_support_tickets.csv", 200)
    
    # Create internal chat logs TXT  
    txt_file = generator.generate_chat_logs_txt("ync_internal_support_chat.txt", 10)
    
    # Create support policy PDF content (saved as TXT for now)
    pdf_content = generator.generate_support_policy_pdf_content()
    with open("ync_support_policies.txt", "w", encoding="utf-8") as f:
        f.write(pdf_content)
    
    print(f"Created ync_support_policies.txt (convert to PDF for assignment)")
    print()
    
    print("Sample Files Created Successfully!")
    print("-" * 40)
    print("Files created:")
    print(f"1. {csv_file} - Customer support tickets with realistic scenarios")
    print(f"2. {txt_file} - Internal team chat logs and communications")  
    print("3. ync_support_policies.txt - Comprehensive support policy document")
    print()
    print("These files are ready to use with your:")
    print("- csv_processor.py for ticket analysis")
    print("- pdf_processor.py for policy document processing") 
    print("- Any text processing for chat log analysis")
    print()
    print("Perfect for testing your Customer Support Triage Agent with Agno!")


if __name__ == "__main__":
    create_all_sample_files()