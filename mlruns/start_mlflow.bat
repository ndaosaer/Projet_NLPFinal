@echo off
REM Script pour lancer MLflow UI
REM À exécuter depuis le dossier racine du projet

echo ================================================================================
echo LANCEMENT DE MLFLOW UI
echo ================================================================================
echo.

echo Verification de l'installation de MLflow...
python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')" 2>nul

if errorlevel 1 (
    echo ERREUR: MLflow n'est pas installe!
    echo.
    echo Installation de MLflow...
    pip install mlflow
    echo.
)

echo ================================================================================
echo DEMARRAGE DU SERVEUR MLFLOW
echo ================================================================================
echo.
echo L'interface MLflow sera accessible sur:
echo   http://localhost:5000
echo.
echo Pour arreter le serveur: Ctrl+C
echo.
echo IMPORTANT: Laissez cette fenetre ouverte pendant que vous travaillez!
echo ================================================================================
echo.

mlflow ui --port 5000

pause
