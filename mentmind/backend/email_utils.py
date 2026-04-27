"""
Email utilities - send verification and password reset emails
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from decouple import config

EMAIL_HOST          = config("EMAIL_HOST", default="smtp.gmail.com")
EMAIL_PORT          = config("EMAIL_PORT", default=587, cast=int)
EMAIL_HOST_USER     = config("EMAIL_HOST_USER", default="")
EMAIL_HOST_PASSWORD = config("EMAIL_HOST_PASSWORD", default="")
DEFAULT_FROM_EMAIL  = config("DEFAULT_FROM_EMAIL", default="noreply@mentminds.com")
FRONTEND_URL        = config("FRONTEND_URL", default="http://localhost:5500")


def send_email(to_email: str, subject: str, html_content: str):
    """
    Send an HTML email.
    In development with no email config, just prints to console.
    """
    if not EMAIL_HOST_USER or not EMAIL_HOST_PASSWORD:
        # Dev mode - print to console
        print(f"\n{'='*60}")
        print(f"📧 EMAIL (dev mode - not actually sent)")
        print(f"To:      {to_email}")
        print(f"Subject: {subject}")
        print(f"{'='*60}\n")
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = DEFAULT_FROM_EMAIL
        msg["To"]      = to_email
        msg.attach(MIMEText(html_content, "html"))

        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
            server.sendmail(DEFAULT_FROM_EMAIL, to_email, msg.as_string())

        return True
    except Exception as e:
        print(f"❌ Email send failed: {e}")
        return False


def send_verification_email(to_email: str, full_name: str, token: str):
    """Send email verification link"""
    verify_url = f"{FRONTEND_URL}/verify-email.html?token={token}"

    # Also print token to console so you can test without email
    print(f"\n✅ VERIFICATION TOKEN for {to_email}: {token}")
    print(f"🔗 Verify URL: {verify_url}\n")

    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #0f172a; color: white; padding: 40px; border-radius: 12px;">
        <h1 style="color: #667eea;">Welcome to MentMinds! 🎓</h1>
        <p>Hi {full_name},</p>
        <p>Thanks for registering. Please verify your email to get started.</p>
        <a href="{verify_url}"
           style="display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2);
                  color: white; padding: 14px 28px; border-radius: 8px; text-decoration: none;
                  font-weight: bold; margin: 20px 0;">
            Verify Email
        </a>
        <p style="color: #6b7280; font-size: 14px;">
            Link expires in 24 hours.<br>
            If you did not sign up, ignore this email.
        </p>
    </div>
    """
    return send_email(to_email, "Verify your MentMinds email", html)


def send_password_reset_email(to_email: str, full_name: str, token: str):
    """Send password reset link"""
    reset_url = f"{FRONTEND_URL}/reset-password.html?token={token}"

    # Print to console for easy testing
    print(f"\n🔑 RESET TOKEN for {to_email}: {token}")
    print(f"🔗 Reset URL: {reset_url}\n")

    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #0f172a; color: white; padding: 40px; border-radius: 12px;">
        <h1 style="color: #667eea;">Password Reset 🔑</h1>
        <p>Hi {full_name},</p>
        <p>We received a request to reset your password.</p>
        <a href="{reset_url}"
           style="display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2);
                  color: white; padding: 14px 28px; border-radius: 8px; text-decoration: none;
                  font-weight: bold; margin: 20px 0;">
            Reset Password
        </a>
        <p style="color: #6b7280; font-size: 14px;">
            Link expires in 1 hour.<br>
            If you did not request this, ignore this email.
        </p>
    </div>
    """
    return send_email(to_email, "Reset your MentMinds password", html)
