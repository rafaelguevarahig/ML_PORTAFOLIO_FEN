# 📊 Optimización de Portafolio de Inversión con Machine Learning

## 🎯 Objetivo del Proyecto

Este proyecto **optimiza la distribución de un portafolio de inversión en acciones** utilizando técnicas financieras cuantitativas y machine learning para maximizar retornos ajustados por riesgo.

---

## 💡 ¿Qué Problema Resuelve?

### Situación Inicial
Tienes un portafolio con **$9,000 invertidos** en 10 empresas diferentes:

```
Apple (AAPL):              $1,500 (16.67%)
Johnson & Johnson (JNJ):   $1,200 (13.33%)
JPMorgan (JPM):            $1,300 (14.44%)
Procter & Gamble (PG):     $800   (8.89%)
Exxon Mobil (XOM):         $700   (7.78%)
3M Company (MMM):          $600   (6.67%)
Southern Company (SO):     $500   (5.56%)
Verizon (VZ):              $600   (6.67%)
Nike (NKE):                $1,000 (11.11%)
DuPont (DD):               $800   (8.89%)
```

### Pregunta Clave
**¿Esta distribución es óptima para maximizar ganancias y minimizar riesgo?**

Este proyecto responde esa pregunta mediante análisis cuantitativo.

---

## 🔬 Metodología: Las 3 Estrategias Comparadas

### 1️⃣ Portafolio Original (Línea Base)
- Distribución inicial sin optimización
- Sirve como referencia para medir mejoras

### 2️⃣ Optimización Mean-Variance (MV)
- Basada en la **Teoría Moderna de Portafolios** de Markowitz (1952)
- Busca el mejor balance entre:
  - ✅ Maximizar retornos esperados
  - ✅ Minimizar riesgo (volatilidad)
- Usa **únicamente datos históricos** de precios

### 3️⃣ Optimización ML + Mean-Variance
- Combina optimización MV con **predicciones de Machine Learning**
- El ML analiza patrones históricos (momentum, tendencias, volatilidad)
- Genera expectativas ajustadas de retorno para cada acción
- Integra estas predicciones mediante el **Modelo Black-Litterman**

---

## 📈 ¿Qué Hace el Proyecto?

### ✅ LO QUE SÍ HACE

1. **Valida Estrategias con Datos Reales**
   - Entrena modelos con datos históricos (2013-2018)
   - Prueba el desempeño en periodo pasado conocido (2018-2023)
   - Compara resultados reales vs mercado (S&P 500)

2. **Genera Pesos Optimizados**
   ```
   ANTES:  AAPL 16.67% | JPM 14.44% | JNJ 13.33%
   DESPUÉS: AAPL 18.23% | JPM 16.78% | JNJ 13.45%
   ```

3. **Calcula Métricas Financieras**
   - Sharpe Ratio (retorno ajustado por riesgo)
   - Sortino Ratio (retorno ajustado por riesgo negativo)
   - Information Ratio (exceso de retorno vs mercado)

4. **Proporciona Recomendaciones Accionables**
   - Comprar más de X acción
   - Vender Y acción
   - Rebalancear portafolio

### ❌ LO QUE NO HACE

- ❌ **NO predice precios futuros** ("Apple subirá a $200 en 2026")
- ❌ **NO hace forecasting tradicional** de series temporales
- ❌ **NO garantiza ganancias futuras**
- ❌ **NO es un sistema de trading automático**

### 🎯 Enfoque Real

```
En lugar de: "Apple subirá 20% en 2026"

El proyecto dice: "Basado en 5 años de datos históricos, 
un portafolio optimizado con ML habría generado 65% de retorno 
vs 38% sin optimización. Aquí están los pesos sugeridos."
```

---

## 🧠 Componentes Técnicos

### 1. **Mean-Variance Optimization** (`mean_variance_optimization.py`)
- Implementa optimización de Markowitz
- Resuelve problema de programación cuadrática
- Restricciones:
  - Volatilidad máxima: 22.5%
  - Peso mínimo por acción: 1%
  - Peso máximo por acción: 25%

### 2. **Machine Learning Strategies** (`machine_learning_strategies.py`)
- Genera predicciones de retorno por acción
- Analiza:
  - Momentum (tendencias de precio)
  - Volatilidad histórica
  - Patrones de comportamiento
- Calcula nivel de confianza en cada predicción

### 3. **Black-Litterman Model** (`black_litterman_model.py`)
- Combina equilibrio de mercado con predicciones ML
- Ajusta expectativas de retorno
- Incorpora capitalización de mercado
- Balancea views del inversor con realidad del mercado

### 4. **Portfolio Statistics** (`portfolio_statistics.py`)
- Calcula métricas de desempeño:
  - **Sharpe Ratio**: Retorno por unidad de riesgo
  - **Sortino Ratio**: Retorno por unidad de riesgo negativo
  - **Information Ratio**: Exceso de retorno vs benchmark

---

## 📊 Métricas de Evaluación

### Sharpe Ratio
```
Fórmula: (Retorno - Tasa Libre Riesgo) / Volatilidad

Interpretación:
- Sharpe = 1.5 → Ganas 1.5% extra por cada 1% de riesgo
- Mayor es mejor
- Valores típicos: 0.5-2.0
```

### Sortino Ratio
```
Similar al Sharpe pero solo penaliza volatilidad negativa
Ignora fluctuaciones al alza (que son positivas)
Mejor para evaluar riesgo de pérdidas
```

### Information Ratio
```
Mide exceso de retorno vs mercado (SPY)
IR > 0 → Superas al mercado
IR = 0.5 es considerado bueno
```

---

## 🚀 Cómo Usar el Proyecto

### 1. Instalación de Dependencias
```bash
cd capstone/ML-Portfolio-Optimization
pip install -r requirements.txt
```

### 2. Configurar Tu Portafolio
Editar `main.py`:
```python
portfolio = {
    'AAPL': 1500.0,  # Tu inversión en cada acción
    'JNJ': 1200.0,
    # ... más acciones
}
```

### 3. Configurar Parámetros
```python
# Fechas de entrenamiento
training_start_date = '2013-11-27'
training_end_date = '2018-11-27'

# Fechas de backtesting (validación)
backtesting_start_date = '2018-11-27'
backtesting_end_date = '2023-11-27'

# Restricciones de riesgo
max_volatility = 0.225  # Máximo 22.5% volatilidad anual
min_weight = 0.01       # Mínimo 1% por acción
max_weight = 0.25       # Máximo 25% por acción
risk_free_rate = 0.04   # Tasa libre de riesgo (bonos)
```

### 4. Ejecutar
```bash
python main.py
```

---

## 📖 Interpretación de Resultados

### Salida en Consola
```
Portfolio Comparison:
                Original    MV Optimization    ML MV Optimization
AAPL            16.67%         12.45%             18.23%  ⬆️ Aumentar
JNJ             13.33%         15.67%             13.45%  ➡️ Mantener
JPM             14.44%         11.34%             16.78%  ⬆️ Aumentar
XOM              7.78%         10.23%              5.12%  ⬇️ Reducir
```

### Gráfica Generada
- **Eje X**: Tiempo (2018-2023)
- **Eje Y**: Retorno acumulado en %
- **4 Líneas**:
  1. ML + MV Optimizado (mejor esperado)
  2. MV Optimizado
  3. Mercado (S&P 500 via SPY)
  4. Portafolio Original

### Cajas de Estadísticas
```
ML & MV Optimized Portfolio:
Sharpe Ratio:  1.45
Sortino Ratio: 2.12
Info Ratio:    0.67
Return:        65.23%  ✅ Mejor resultado
```

---

## 💼 Caso de Uso Práctico

### Escenario: Inversionista Rebalanceando Portafolio

**Paso 1**: Ejecutar análisis con portafolio actual
```bash
python main.py
```

**Paso 2**: Revisar recomendaciones
```
ML MV sugiere:
- Aumentar AAPL de 16.67% → 18.23% (+$140)
- Aumentar JPM de 14.44% → 16.78% (+$210)
- Reducir XOM de 7.78% → 5.12% (-$240)
```

**Paso 3**: Evaluar evidencia histórica
```
Si hubieras aplicado esta estrategia en 2018:
Retorno: 65.23% vs 38.45% sin optimizar
Diferencia: +26.78 puntos porcentuales
```

**Paso 4**: Tomar decisión informada
```
Basado en 5 años de evidencia:
✅ Aplicar optimización tiene fundamento estadístico
✅ Estrategia ha superado al mercado históricamente
⚠️ Rendimientos pasados no garantizan futuros
```

**Paso 5**: Ejecutar órdenes de rebalanceo
```
1. Vender $240 de XOM
2. Comprar $140 adicionales de AAPL
3. Comprar $210 adicionales de JPM
```

---

## ⚠️ Limitaciones y Consideraciones

### Limitaciones del Modelo

1. **No es predictivo a futuro**
   - Valida estrategias con datos históricos
   - No predice precios futuros específicos

2. **Asume continuidad de patrones**
   - Los patrones históricos pueden cambiar
   - Crisis o eventos disruptivos no son predecibles

3. **Costos de transacción no incluidos**
   - Comisiones de compra/venta
   - Impuestos sobre ganancias
   - Spreads bid-ask

4. **Datos limitados**
   - Depende de calidad de datos históricos
   - Yahoo Finance puede tener errores

### Consideraciones para el Inversor

✅ **Usar como herramienta de apoyo**, no decisión única  
✅ **Rebalancear periódicamente** (trimestral/semestral)  
✅ **Considerar tu perfil de riesgo** personal  
✅ **Diversificar más allá de acciones** (bonos, real estate)  
✅ **Consultar asesor financiero** para decisiones importantes  

❌ **NO confiar ciegamente** en cualquier modelo  
❌ **NO invertir dinero que no puedes perder**  
❌ **NO ignorar fundamentals** de las empresas  

---

## 🔧 Personalización Avanzada

### Ajustar Tolerancia al Riesgo

**Conservador** (bajo riesgo):
```python
max_volatility = 0.15  # Solo 15% volatilidad
min_weight = 0.05      # Más diversificado
max_weight = 0.20      # Menos concentración
```

**Moderado** (riesgo medio):
```python
max_volatility = 0.225  # 22.5% volatilidad
min_weight = 0.01
max_weight = 0.25
```

**Agresivo** (alto riesgo):
```python
max_volatility = 0.35   # Acepto 35% volatilidad
min_weight = 0.01
max_weight = 0.40       # Permito concentración
```

### Cambiar Benchmark
```python
# S&P 500
market_representation = ['SPY']

# NASDAQ 100
market_representation = ['QQQ']

# Russell 2000
market_representation = ['IWM']
```

### Rebalanceo Automático
Para uso regular, ejecutar cada trimestre:
```bash
# Cron job (Linux/Mac)
0 0 1 */3 * cd /path/to/project && python main.py
```

---

## 📚 Fundamentos Teóricos

### Teoría Moderna de Portafolios (Markowitz, 1952)
- Premio Nobel de Economía 1990
- Concepto de **frontera eficiente**
- Diversificación reduce riesgo sin sacrificar retorno

### Black-Litterman Model (1992)
- Desarrollado por Goldman Sachs
- Combina equilibrio de mercado con views del inversor
- Evita estimaciones extremas de Mean-Variance puro

### Métricas de Sharpe/Sortino
- Sharpe Ratio (1966): William F. Sharpe, Premio Nobel 1990
- Sortino Ratio (1980s): Mejora considerando solo downside risk

---

## 📁 Estructura del Proyecto

```
ML-Portfolio-Optimization/
│
├── main.py                          # Script principal
├── mean_variance_optimization.py    # Optimización Markowitz
├── machine_learning_strategies.py   # Modelos ML
├── black_litterman_model.py         # Modelo Black-Litterman
├── portfolio_statistics.py          # Cálculo de métricas
├── factor_analysis.py               # (Para uso futuro)
├── requirements.txt                 # Dependencias Python
├── README.md                        # Documentación original
└── EXPLICACION_PROYECTO.md          # Esta guía
```

---

## 🎓 Conclusión

Este proyecto proporciona una **metodología cuantitativamente rigurosa** para optimizar portafolios de inversión, combinando:

1. ✅ Teoría financiera probada (70+ años)
2. ✅ Machine Learning moderno
3. ✅ Validación con datos reales
4. ✅ Métricas estándar de la industria

**Valor Principal**: No adivina el futuro, sino que aplica estrategias que **han demostrado funcionar** en periodos históricos extensos.

**Uso Recomendado**: Herramienta de apoyo para toma de decisiones informadas en gestión de portafolios, complementando análisis fundamental y asesoría profesional.

---

## 📞 Información Adicional

**Autor**: Rafael Guevara  
**Contexto**: Diplomado en Ciencia de Datos e Inteligencia Artificial  
**Fecha**: Enero 2026  

**Tecnologías**: Python, NumPy, Pandas, Matplotlib, Scikit-learn, yfinance

---

## ⚖️ Disclaimer

Este proyecto es solo para fines educativos y de investigación. No constituye asesoría financiera profesional. Las decisiones de inversión deben tomarse consultando con asesores financieros certificados y considerando tu situación personal completa.

**Rendimientos pasados no garantizan resultados futuros.**
