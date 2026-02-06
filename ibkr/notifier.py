"""
Trade notification system for IBKR trading.

Sends email and SMS alerts for trade executions.
Credentials are loaded from environment variables for security.
"""

import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class NotificationConfig:
    """Configuration for trade notifications."""
    # Email settings
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    notify_email: Optional[str] = None

    # SMS settings (Twilio)
    twilio_sid: Optional[str] = None
    twilio_token: Optional[str] = None
    twilio_from: Optional[str] = None
    notify_phone: Optional[str] = None

    @property
    def email_enabled(self) -> bool:
        """Check if email notifications are configured."""
        return all([
            self.smtp_server,
            self.smtp_user,
            self.smtp_password,
            self.notify_email
        ])

    @property
    def sms_enabled(self) -> bool:
        """Check if SMS notifications are configured."""
        return all([
            self.twilio_sid,
            self.twilio_token,
            self.twilio_from,
            self.notify_phone
        ])

    @classmethod
    def from_env(cls) -> "NotificationConfig":
        """Load configuration from environment variables."""
        return cls(
            smtp_server=os.getenv("IBKR_SMTP_SERVER"),
            smtp_port=int(os.getenv("IBKR_SMTP_PORT", "587")),
            smtp_user=os.getenv("IBKR_SMTP_USER"),
            smtp_password=os.getenv("IBKR_SMTP_PASSWORD"),
            notify_email=os.getenv("IBKR_NOTIFY_EMAIL"),
            twilio_sid=os.getenv("IBKR_TWILIO_SID"),
            twilio_token=os.getenv("IBKR_TWILIO_TOKEN"),
            twilio_from=os.getenv("IBKR_TWILIO_FROM"),
            notify_phone=os.getenv("IBKR_NOTIFY_PHONE"),
        )


class TradeNotifier:
    """Sends trade notifications via email and SMS."""

    def __init__(self, config: Optional[NotificationConfig] = None):
        """
        Initialize notifier.

        Args:
            config: Notification configuration. If None, loads from env vars.
        """
        self.config = config or NotificationConfig.from_env()
        self._twilio_client = None

    def _get_twilio_client(self):
        """Lazy-load Twilio client."""
        if self._twilio_client is None and self.config.sms_enabled:
            try:
                from twilio.rest import Client
                self._twilio_client = Client(
                    self.config.twilio_sid,
                    self.config.twilio_token
                )
            except ImportError:
                logger.warning("Twilio package not installed. SMS disabled.")
                return None
        return self._twilio_client

    def notify_trade(
        self,
        action: str,
        symbol: str,
        shares: int,
        price: float,
        status: str,
        order_id: Optional[int] = None,
        account: Optional[str] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Send trade notification via all configured channels.

        Args:
            action: BUY or SELL
            symbol: Stock symbol
            shares: Number of shares
            price: Execution price
            status: Order status (FILLED, SUBMITTED, ERROR, etc.)
            order_id: IBKR order ID
            account: Account ID
            error: Error message if failed

        Returns:
            True if at least one notification was sent successfully
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build message
        subject = f"IBKR Trade: {action} {shares} {symbol} - {status}"

        body_lines = [
            f"Trade Notification",
            f"==================",
            f"Time:    {timestamp}",
            f"Action:  {action}",
            f"Symbol:  {symbol}",
            f"Shares:  {shares}",
            f"Price:   ${price:.2f}",
            f"Value:   ${shares * price:,.2f}",
            f"Status:  {status}",
        ]

        if order_id:
            body_lines.append(f"Order:   {order_id}")
        if account:
            body_lines.append(f"Account: {account}")
        if error:
            body_lines.append(f"Error:   {error}")

        body = "\n".join(body_lines)

        # Short version for SMS
        sms_body = f"IBKR: {action} {shares} {symbol} @ ${price:.2f} = ${shares * price:,.2f} [{status}]"
        if error:
            sms_body += f" ERR: {error[:50]}"

        email_sent = self._send_email(subject, body)
        sms_sent = self._send_sms(sms_body)

        return email_sent or sms_sent

    def _send_email(self, subject: str, body: str) -> bool:
        """Send email notification to all configured recipients."""
        if not self.config.email_enabled:
            logger.debug("Email notifications not configured")
            return False

        # Support comma-separated emails — send individually so recipients don't see each other
        recipients = [e.strip() for e in self.config.notify_email.split(",")]

        try:
            sent = []
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                for recipient in recipients:
                    msg = MIMEMultipart()
                    msg["From"] = self.config.smtp_user
                    msg["To"] = recipient
                    msg["Subject"] = subject
                    msg.attach(MIMEText(body, "plain"))
                    server.send_message(msg)
                    sent.append(recipient)

            logger.info(f"Email sent individually to {len(sent)} recipient(s): {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email ({len(sent)}/{len(recipients)} sent): {e}")
            return False

    def _send_sms(self, body: str) -> bool:
        """Send SMS notification via Twilio."""
        if not self.config.sms_enabled:
            logger.debug("SMS notifications not configured")
            return False

        client = self._get_twilio_client()
        if client is None:
            return False

        try:
            message = client.messages.create(
                body=body,
                from_=self.config.twilio_from,
                to=self.config.notify_phone
            )
            logger.info(f"SMS sent: {message.sid}")
            return True

        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False

    def test_notifications(self) -> dict:
        """
        Test notification channels.

        Returns:
            Dict with status of each channel
        """
        results = {
            "email_configured": self.config.email_enabled,
            "sms_configured": self.config.sms_enabled,
            "email_sent": False,
            "sms_sent": False,
        }

        if self.config.email_enabled:
            results["email_sent"] = self._send_email(
                "IBKR Test Notification",
                "This is a test notification from your IBKR trading system."
            )

        if self.config.sms_enabled:
            results["sms_sent"] = self._send_sms(
                "IBKR Test: Notifications working!"
            )

        return results