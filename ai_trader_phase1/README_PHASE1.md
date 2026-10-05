# AI Trader - Fase 1: Automatización local de pruebas

## Objetivo

Agregar una capa de automatización para Windows/VS Code/Python sin modificar módulos de producción.

### Archivos agregados

- `pytest.ini`
- `requirements-test.txt`
- `test_runner.ps1`
- `tests/conftest.py`
- `tests/phase1/test_repository_structure.py`
- `tests/phase1/test_phase1_safety.py`

### Evidencias

Cada ejecución genera:

```text
evidence/
  YYYYMMDD_HHMMSS/
    test_results.txt
    junit.xml
    pytest_report.html
    git_status_before.txt
    git_status_after.txt
    summary.txt
```

Si se usa `-CollectCoverage`, también:

```text
coverage-html/
```

## Instalación

Desde la raíz del repositorio:

```powershell
python -m pip install -r requirements-test.txt
```

## Ejecución básica

```powershell
.\test_runner.ps1
```

Con cobertura:

```powershell
.\test_runner.ps1 -CollectCoverage
```

Con apertura automática del HTML:

```powershell
.\test_runner.ps1 -OpenReport
```

Ambos:

```powershell
.\test_runner.ps1 -CollectCoverage -OpenReport
```

## Seguridad

La Fase 1:

- no modifica `app/`
- no reemplaza `app/main.py`
- no ejecuta órdenes Alpaca
- no requiere Telegram
- no requiere acceso a Internet para los tests de infraestructura
- genera las evidencias fuera de `app/`

## Siguiente paso

Una vez validado este runner en el repositorio real, se incorporan progresivamente los tests funcionales existentes, sin moverlos ni alterar producción.
