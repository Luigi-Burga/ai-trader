# Matriz de reglas y plan de migración — AI Trader

## 1. Inventario de reglas observado en los módulos entregados

| Criterio | CCE V1.7 | Decision Gate V1.5 | Scanner / Signal Engine / main | Acción propuesta |
|---|---|---|---|---|
| Score | BUY ≥70; STRONG_BUY ≥85 | No es el único criterio de autorización | No debería recalcularlo | CCE calcula; política consume |
| Probabilidad 20d | BUY ≥0.60; STRONG_BUY ≥0.65 | CCE Gate ≥0.60; vía alternativa requiere ≥0.65 | No repetir | Una política operativa |
| R/R | BUY ≥1.25; STRONG_BUY ≥2.0 | CCE Gate ≥1.25; alternativa ≥1.50 | Ejecutor comprueba riesgo real | Separar elegibilidad y riesgo de orden |
| Confianza | Degradación bajo 0.50; decisión operativa exige 0.60 | Vía principal ≥0.60; alternativa ≥0.50 con requisitos reforzados | No crear tercer umbral de calidad | Consolidar dos rutas explícitas |
| Fallback | Señal analítica tolera hasta 0.50; decisión operativa limita a 0.25 | CCE Gate usa límite 0.50 | No volver a evaluar | Política operativa: ≤0.25 para automático; ≤0.50 solo para confirmación |
| Calidad histórica | Calidad débil puede degradar a WATCH | Principal GOOD+; alternativa MODERATE con condiciones reforzadas | No repetir | Una decisión única |
| Benchmark | CCE evalúa confianza del benchmark | Gate valida trust/decision grade | Signal Engine/Scanner pueden degradar a BUY_ON_CONFIRMATION | Una validación central; conservar el dato |
| Régimen | Lista de ciclos bloqueados | Vuelve a evaluar régimen | Scanner conserva señales no alcistas | Lista única y versionada |
| Entrada/stop | Calcula niveles dinámicos | Comprueba geometría y validez | Ejecutor debe verificar precio/stop/riesgo vigentes | Gate valida oportunidad; ejecutor valida orden |
| Señal final | Puede degradar BUY | Puede degradar a BUY_ON_CONFIRMATION | Scanner evita revertir señal; main ejecuta BUY/STRONG_BUY | Solo política asigna decisión operativa |
| Seguridad de ejecución | No ejecuta | No envía orden | `main.py`/executor controla ejecución | Mantener independiente y obligatorio |

## 2. Política propuesta
- **BUY automático elegible:** score ≥70, probabilidad 20d ≥0.60, R/R ≥1.25, confianza ≥0.60, fallback ≤0.25, calidad GOOD o superior, benchmark apto, régimen permitido, datos y entrada/stop válidos.
- **STRONG_BUY:** los criterios anteriores más score ≥85, probabilidad ≥0.65 y R/R ≥2.0.
- **BUY_ON_CONFIRMATION:** no es una orden automática. Se permite solo si score ≥70, probabilidad ≥0.65, R/R ≥1.50, confianza ≥0.50, fallback ≤0.50, calidad MODERATE o superior, APPT ≥0.50, benchmark y datos aptos, y régimen permitido.
- **WATCH / REVIEW:** cualquier candidato que no pase las rutas anteriores.
- **SELL / REDUCE / AVOID:** preservar; la política no debe convertirlos en compras.
- **Ejecución:** incluso BUY/STRONG_BUY deben superar las protecciones existentes del ejecutor. Esta política no sustituye riesgo por operación, exposición máxima, cuenta activa, órdenes duplicadas ni autorización.

## 3. Diferencias deliberadas frente al comportamiento distribuido
1. El umbral de fallback para compra automática se centraliza en 0.25; 0.50 solo puede habilitar confirmación bajo condiciones reforzadas.
2. Confianza 0.50 no basta por sí sola: solo aplica a la vía de confirmación con todos sus criterios.
3. El benchmark que solo aporta contexto nunca habilita compra automática.
4. Los regímenes bloqueados se evalúan una sola vez en la política; los controles del ejecutor siguen independientes.

## 4. Secuencia segura
1. Ejecutar pruebas unitarias de la política aislada.
2. Implementar adaptador de salida CCE/Decision Gate sin cambiar decisiones de producción.
3. Ejecutar modo sombra con los mismos datos: decisión actual vs. decisión propuesta por símbolo.
4. Revisar diferencias y ajustar reglas aprobadas.
5. Solo después, integrar la política en el Scanner y retirar las evaluaciones duplicadas.
6. Regresión completa y Paper; nunca activar ejecución automática durante el modo sombra.

## 5. Riesgo de migración
La matriz describe reglas de los archivos entregados; todavía no prueba qué rama se ejecutó para cada símbolo en producción. La política aislada no debe conectarse al ejecutor hasta completar el modo sombra.
