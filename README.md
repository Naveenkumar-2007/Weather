# 🌦️ Weather Prediction Web App (Flask + ML Pipeline)

This project predicts **rainfall/weather conditions** using a **Machine Learning pipeline** wrapped in a **Flask web application**.  
The user inputs weather-related features through a web form, and the model predicts whether it will rain tomorrow.  

---

## 🚀 Features  

- 🔹 **Flask Web Application** with HTML front-end (`h.html`, `home.html`)  
- 🔹 Accepts user input for multiple weather parameters  
- 🔹 Data processed using a **custom pipeline (`customdata`, `predict`)**  
- 🔹 Machine Learning model predicts **rainfall outcome (1 = Rain Tomorrow, 0 = No Rain Tomorrow)**  
- 🔹 Displays results on the web interface  

---

## 📊 Input and Output  

### 🔹 Input Features (form fields)  

| Feature        | Type   | Description |
|----------------|--------|-------------|
| **MinTemp**    | float  | Minimum temperature (°C) |
| **MaxTemp**    | float  | Maximum temperature (°C) |
| **Rainfall**   | float  | Rainfall amount (mm) |
| **Evaporation**| float  | Evaporation level |
| **Humidity9am**| float  | Humidity at 9 AM (%) |
| **Humidity3pm**| float  | Humidity at 3 PM (%) |
| **Pressure9am**| float  | Atmospheric pressure at 9 AM (hPa) |
| **Pressure3pm**| float  | Atmospheric pressure at 3 PM (hPa) |
| **Cloud9am**   | float  | Cloud coverage at 9 AM (oktas) |
| **Cloud3pm**   | float  | Cloud coverage at 3 PM (oktas) |
| **Temp9am**    | float  | Temperature at 9 AM (°C) |
| **Temp3pm**    | float  | Temperature at 3 PM (°C) |
| **RISK_MM**    | float  | Rainfall risk factor |
| **RainToday**  | string | "Yes" or "No" (whether it rained today) |

---

### 🔹 Output  

- The ML model predicts **Rain Tomorrow (binary classification)**:  

| Value | Meaning |
|-------|---------|
| **1** | 🌧️ Rain Tomorrow |
| **0** | ☀️ No Rain Tomorrow |

- Example output shown on webpage:  

```html
Prediction: 1 → Rain Tomorrow
Prediction: 0 → No Rain Tomorrow
