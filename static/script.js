document.addEventListener('DOMContentLoaded', () => {
    // ---------------------------------------------------------
    // 1. Tab Switching Logic
    // ---------------------------------------------------------
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.add('hidden'));

            btn.classList.add('active');
            document.getElementById(targetTab).classList.remove('hidden');

            if (targetTab === 'model-metrics-tab') {
                loadModelMetrics();
            }
        });
    });

    // ---------------------------------------------------------
    // 2. Single Churn Prediction Form Handler
    // ---------------------------------------------------------
    const churnForm = document.getElementById('churnForm');
    const predictBtn = document.getElementById('predictBtn');
    const placeholderView = document.getElementById('placeholderView');
    const resultsView = document.getElementById('resultsView');

    if (churnForm) {
        churnForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Button loading state
            predictBtn.disabled = true;
            predictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Calculating Risk...';

            const formData = new FormData(churnForm);
            const payload = {};
            formData.forEach((value, key) => {
                payload[key] = value;
            });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                if (data.status === 'success') {
                    renderResults(data);
                } else {
                    alert('Error: ' + (data.message || data.error));
                }
            } catch (err) {
                console.error(err);
                alert('Failed to connect to prediction server.');
            } finally {
                predictBtn.disabled = false;
                predictBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> Predict Churn & Risk';
            }
        });
    }

    function renderResults(data) {
        placeholderView.classList.add('hidden');
        resultsView.classList.remove('hidden');

        const riskTag = document.getElementById('riskTag');
        const probScore = document.getElementById('probScore');
        const meterBarFill = document.getElementById('meterBarFill');
        const actionText = document.getElementById('actionText');
        const recsList = document.getElementById('recsList');
        const driversList = document.getElementById('driversList');

        // Set Risk Tag and Color
        riskTag.textContent = data.risk_level.toUpperCase();
        riskTag.style.backgroundColor = data.risk_color;

        // Set Probability Score & Meter
        const prob = data.churn_probability;
        probScore.textContent = `${prob}%`;
        meterBarFill.style.width = `${prob}%`;
        meterBarFill.style.backgroundColor = data.risk_color;

        // Action Text
        actionText.textContent = data.risk_action;

        // Recommendations List
        recsList.innerHTML = '';
        if (data.recommendations && data.recommendations.length > 0) {
            data.recommendations.forEach(rec => {
                const li = document.createElement('li');
                li.innerHTML = rec;
                recsList.appendChild(li);
            });
        }

        // Drivers List
        driversList.innerHTML = '';
        if (data.top_drivers && data.top_drivers.length > 0) {
            data.top_drivers.forEach(driver => {
                if (driver.Contribution > 0) {
                    const div = document.createElement('div');
                    div.className = 'driver-item';
                    div.innerHTML = `
                        <span class="driver-name">${driver.Feature}</span>
                        <span class="driver-contrib">+${driver.Contribution.toFixed(2)} log-odds</span>
                    `;
                    driversList.appendChild(div);
                }
            });
        }
    }

    // ---------------------------------------------------------
    // 3. Load Model Evaluation Metrics
    // ---------------------------------------------------------
    async function loadModelMetrics() {
        const tbody = document.getElementById('metricsTableBody');
        if (!tbody) return;

        try {
            const response = await fetch('/api/metrics');
            const data = await response.json();

            if (data.metrics) {
                tbody.innerHTML = '';
                const bestModel = data.best_model;

                for (const [modelName, m] of Object.entries(data.metrics)) {
                    const tr = document.createElement('tr');
                    const isBest = modelName === bestModel;
                    tr.innerHTML = `
                        <td><strong>${modelName}</strong></td>
                        <td>${(m.Accuracy * 100).toFixed(2)}%</td>
                        <td>${(m.Precision * 100).toFixed(2)}%</td>
                        <td>${(m.Recall * 100).toFixed(2)}%</td>
                        <td>${(m['F1 Score'] * 100).toFixed(2)}%</td>
                        <td>${(m['ROC AUC'] * 100).toFixed(2)}%</td>
                        <td>${isBest ? '<span class="badge-best">🏆 Best Model</span>' : '<span style="color:#64748b;">Evaluated</span>'}</td>
                    `;
                    tbody.appendChild(tr);
                }
            }
        } catch (err) {
            tbody.innerHTML = '<tr><td colspan="7" style="color:red;">Failed to load metrics. Ensure model is trained.</td></tr>';
        }
    }

    // ---------------------------------------------------------
    // 4. Batch CSV Upload & Drag and Drop Handler
    // ---------------------------------------------------------
    const batchFile = document.getElementById('batchFile');
    const dropZone = document.getElementById('dropZone');

    if (dropZone && batchFile) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('highlight'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('highlight'), false);
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length > 0) {
                batchFile.files = files;
                handleFileUpload(files[0]);
            }
        });

        batchFile.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });
    }

    async function handleFileUpload(file) {
        if (!file.name.endsWith('.csv')) {
            alert('Please select a valid CSV file (.csv)');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        // Show uploading UI indicator
        const dropText = dropZone.querySelector('p');
        const originalText = dropText.innerHTML;
        dropText.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing ${file.name}...`;

        try {
            const response = await fetch('/api/batch_predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (data.status === 'success') {
                dropText.innerHTML = `<i class="fa-solid fa-circle-check" style="color:#28a745;"></i> Processed: <strong>${file.name}</strong> (${data.summary.total_records} records)`;
                renderBatchResults(data);
            } else {
                dropText.innerHTML = originalText;
                alert('Error: ' + (data.message || data.error));
            }
        } catch (err) {
            console.error(err);
            dropText.innerHTML = originalText;
            alert('Batch prediction request failed. Check server console.');
        }
    }

    function renderBatchResults(data) {
        const batchResults = document.getElementById('batchResults');
        const statsGrid = document.getElementById('statsGrid');
        const previewBody = document.getElementById('batchPreviewBody');

        batchResults.classList.remove('hidden');
        statsGrid.innerHTML = `
            <div class="card" style="padding:16px; text-align:center;">
                <h4>Total Scored</h4>
                <p style="font-size:1.8rem; font-weight:800; color:#0f172a;">${data.summary.total_records || 0}</p>
            </div>
            <div class="card" style="padding:16px; text-align:center;">
                <h4>Avg Churn Rate</h4>
                <p style="font-size:1.8rem; font-weight:800; color:#dc3545;">${(data.summary.churn_rate || 0).toFixed(1)}%</p>
            </div>
            <div class="card" style="padding:16px; text-align:center;">
                <h4>High Risk Customers</h4>
                <p style="font-size:1.8rem; font-weight:800; color:#e11d48;">${data.summary.high_risk_count || 0}</p>
            </div>
        `;

        previewBody.innerHTML = '';
        if (data.preview && data.preview.length > 0) {
            data.preview.forEach(row => {
                const tr = document.createElement('tr');

                const tenureVal = row['Tenure Months'] ?? row['tenure'] ?? row['Tenure'] ?? '-';
                const monthlyVal = row['Monthly Charges'] ?? row['monthly_charges'] ?? row['MonthlyCharges'] ?? '-';
                const contractVal = row['Contract'] ?? row['contract'] ?? '-';
                const internetVal = row['Internet Service'] ?? row['internet_service'] ?? row['InternetService'] ?? '-';
                const probVal = row['Churn_Probability_%'] ?? row['Churn Probability'] ?? '-';
                const riskVal = row['Risk_Level'] ?? row['Risk Level'] ?? '-';

                let riskColor = '#28a745';
                if (riskVal === 'Medium Risk') riskColor = '#d97706';
                if (riskVal === 'High Risk') riskColor = '#dc3545';

                tr.innerHTML = `
                    <td>${tenureVal} mos</td>
                    <td>$${monthlyVal}</td>
                    <td>${contractVal}</td>
                    <td>${internetVal}</td>
                    <td>${typeof probVal === 'number' ? probVal.toFixed(1) + '%' : probVal + '%'}</td>
                    <td><span style="font-weight:700; color:${riskColor};">${riskVal}</span></td>
                `;
                previewBody.appendChild(tr);
            });
        }
    }
});
