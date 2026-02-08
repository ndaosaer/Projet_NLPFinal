@echo off
REM Script de correction automatique pour le problème NumPy
REM À exécuter depuis le dossier racine du projet

echo ================================================================================
echo CORRECTION DU PROBLEME NUMPY 2.x
echo ================================================================================
echo.

echo Etape 1: Desinstallation de NumPy 2.x...
pip uninstall numpy -y

echo.
echo Etape 2: Installation de NumPy 1.26.4 (compatible)...
pip install "numpy==1.26.4"

echo.
echo Etape 3: Reinstallation des dependances compatibles...
pip install --upgrade scikit-learn pandas pyarrow

echo.
echo Etape 4: Verification des versions...
python -c "import numpy; import sklearn; import pandas; print(f'NumPy: {numpy.__version__}'); print(f'scikit-learn: {sklearn.__version__}'); print(f'pandas: {pandas.__version__}')"

echo.
echo ================================================================================
echo CORRECTION TERMINEE!
echo ================================================================================
echo.
echo Vous pouvez maintenant lancer l'API:
echo   cd api
echo   python main.py
echo.
pause
