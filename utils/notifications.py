import os
import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def setup_email_client():
    """
    Set up email client with credentials from environment variables
    
    Returns:
    --------
    tuple
        (smtp_settings, from_email) if credentials are available, (None, None) otherwise
    """
    # Get email credentials from environment variables
    email_address = os.environ.get('EMAIL_ADDRESS')
    email_password = os.environ.get('EMAIL_PASSWORD')
    smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.environ.get('SMTP_PORT', 587))
    
    # Check if credentials are available
    if email_address and email_password:
        smtp_settings = {
            'server': smtp_server,
            'port': smtp_port,
            'user': email_address,
            'password': email_password
        }
        return smtp_settings, email_address
    else:
        return None, None

def send_email_notification(recipient_email, subject, message):
    """
    Send email notification
    
    Parameters:
    -----------
    recipient_email : str
        Recipient email address
    subject : str
        Email subject
    message : str
        Message content to send
        
    Returns:
    --------
    bool
        True if email was sent successfully, False otherwise
    """
    # Set up email client
    smtp_settings, from_email = setup_email_client()
    
    if smtp_settings is None:
        st.error("Email credentials not configured. Please set EMAIL_ADDRESS and EMAIL_PASSWORD environment variables.")
        return False
    
    try:
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Attach message body
        msg.attach(MIMEText(message, 'plain'))
        
        # Connect to SMTP server and send email
        server = smtplib.SMTP(smtp_settings['server'], smtp_settings['port'])
        server.starttls()  # Secure the connection
        server.login(smtp_settings['user'], smtp_settings['password'])
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

def create_race_alert_message(driver_name, team_name, event_name, alert_type, details):
    """
    Create a formatted message for race alerts
    
    Parameters:
    -----------
    driver_name : str
        Driver name or code
    team_name : str
        Team name
    event_name : str
        Event name
    alert_type : str
        Type of alert (e.g., "Fastest Lap", "Pit Stop", "Position Change")
    details : str
        Additional details about the alert
        
    Returns:
    --------
    tuple
        (subject, message) - formatted subject and message for the email
    """
    subject = f"F1 ALERT: {alert_type} - {driver_name} at {event_name}"
    
    message = f"""🏎️ F1 ALERT: {alert_type}

Driver: {driver_name}
Team: {team_name}
Event: {event_name}

{details}

--
Sent by F1 Analytics Platform
"""
    
    return subject, message