const backgroundColors = [
    'rgba(255, 99, 132, 0.7)',     // Sangat Buruk
    'rgba(255, 159, 64, 0.7)',     // Buruk
    'rgba(255, 206, 86, 0.7)',     // Standar
    'rgba(75, 192, 192, 0.7)',     // Baik
    'rgba(54, 162, 235, 0.7)'      // Sangat Baik
];

// Pie Chart
new Chart(document.getElementById('pieChart').getContext('2d'), {
    type: 'pie',
    data: {
        labels: labels,
        datasets: [{
            data: probabilities,
            backgroundColor: backgroundColors,
            borderColor: '#fff',
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                position: 'bottom',
                labels: { color: 'white' }
            },
            tooltip: {
                callbacks: {
                    label: ctx => `${ctx.label}: ${ctx.raw.toFixed(2)}%`
                }
            }
        }
    }
});

// Bar Chart
new Chart(document.getElementById('barChart').getContext('2d'), {
    type: 'bar',
    data: {
        labels: labels,
        datasets: [{
            label: 'Probabilitas (%)',
            data: probabilities,
            backgroundColor: backgroundColors,
            borderColor: backgroundColors.map(c => c.replace('0.7', '1')),
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                ticks: {
                    color: 'white',
                    callback: value => value + '%'
                }
            },
            x: {
                ticks: { color: 'white' }
            }
        },
        plugins: {
            legend: {
                labels: { color: 'white' }
            }
        }
    }
});
