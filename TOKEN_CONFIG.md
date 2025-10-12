# Cấu hình Token Authentication

## ⏰ Thời gian sống Token

### Access Token
- **Thời gian sống:** 1 ngày (1440 phút)
- **Mục đích:** Token chính để xác thực các API requests
- **Khi hết hạn:** Tự động sử dụng refresh token để lấy token mới

### Refresh Token
- **Thời gian sống:** 30 ngày
- **Mục đích:** Dùng để làm mới access token khi hết hạn
- **Khi hết hạn:** User phải đăng nhập lại

## 🔧 Cấu hình

File: `backend/app/core/config.py`

```python
ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 1 day (24 hours * 60 minutes)
REFRESH_TOKEN_EXPIRE_DAYS: int = 30      # 30 days
```

## 🔄 Luồng xác thực

1. **Login thành công:**
   - Nhận access token (sống 1 ngày)
   - Nhận refresh token (sống 30 ngày)
   - Cả 2 token được lưu trong AsyncStorage

2. **Khi access token hết hạn:**
   - App tự động phát hiện
   - Sử dụng refresh token để lấy access token mới
   - Không cần user đăng nhập lại

3. **Khi cả 2 token hết hạn:**
   - App tự động logout
   - Chuyển về màn hình đăng nhập
   - User cần đăng nhập lại

## 📝 Lợi ích

✅ **Access token 1 ngày:**
- User không phải đăng nhập lại mỗi ngày
- Vẫn đảm bảo bảo mật (token ngắn hạn)
- UX tốt hơn cho ứng dụng di động

✅ **Refresh token 30 ngày:**
- User chỉ cần đăng nhập lại mỗi tháng
- Có thời gian đủ dài để sử dụng app thoải mái
- Vẫn đảm bảo bảo mật (phải đăng nhập định kỳ)

## 🛠️ Tùy chỉnh

Để thay đổi thời gian sống token:

1. Mở file `backend/app/core/config.py`
2. Sửa giá trị:
   - `ACCESS_TOKEN_EXPIRE_MINUTES`: Đơn vị phút
   - `REFRESH_TOKEN_EXPIRE_DAYS`: Đơn vị ngày
3. Restart backend server

**Ví dụ:**
```python
# Access token sống 2 ngày
ACCESS_TOKEN_EXPIRE_MINUTES: int = 2880  # 2 * 24 * 60

# Refresh token sống 60 ngày
REFRESH_TOKEN_EXPIRE_DAYS: int = 60
```

## 🔐 Best Practices

- ✅ Access token nên có thời gian ngắn (vài giờ đến 1 ngày)
- ✅ Refresh token nên dài hơn (vài ngày đến vài tháng)
- ✅ Không lưu token trong localStorage (dùng AsyncStorage/SecureStore)
- ✅ Tự động refresh token khi access token hết hạn
- ✅ Logout user khi refresh token hết hạn

---

**Cập nhật lần cuối:** October 11, 2025
