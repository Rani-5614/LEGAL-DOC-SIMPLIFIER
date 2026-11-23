// static/js/main.js

document.addEventListener('DOMContentLoaded', function() {
    // Auto-dismiss alerts after 5 seconds
    setTimeout(function() {
        const alerts = document.querySelectorAll('.alert-dismissible');
        alerts.forEach(function(alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);

    // File upload preview
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const fileName = file.name;
                const fileSize = (file.size / 1024).toFixed(2);
                const preview = document.getElementById('filePreview');
                if (preview) {
                    preview.innerHTML = `
                        <i class="fas fa-file me-2"></i>
                        ${fileName} (${fileSize} KB)
                    `;
                }
            }
        });
    }

    // Form validation
    const forms = document.querySelectorAll('form[data-validate]');
    forms.forEach(function(form) {
        form.addEventListener('submit', function(e) {
            const requiredFields = form.querySelectorAll('[required]');
            let valid = true;
            
            requiredFields.forEach(function(field) {
                if (!field.value.trim()) {
                    field.classList.add('is-invalid');
                    valid = false;
                } else {
                    field.classList.remove('is-invalid');
                }
            });
            
            if (!valid) {
                e.preventDefault();
            }
        });
    });

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Helper function to format dates
function formatDate(dateString) {
    const date = new Date(dateString);
    const options = { 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    };
    return date.toLocaleDateString('en-IN', options);
}

// Copy to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showToast('Copied to clipboard!', 'success');
    }, function(err) {
        showToast('Failed to copy', 'danger');
    });
}

// Toast notification
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toastContainer') || createToastContainer();
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    toastContainer.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toastContainer';
    container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
    document.body.appendChild(container);
    return container;
}

// ------------------------------------------------------------
// Deadline reminders: banner + optional browser notifications
// ------------------------------------------------------------
function escapeHtml(str) {
    return String(str == null ? '' : str).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}

document.addEventListener('DOMContentLoaded', function() {
    if (!window.LEGALEASE_AUTH) return;
    const banner = document.getElementById('reminderBanner');
    fetch('/api/upcoming_deadlines')
        .then(r => r.ok ? r.json() : null)
        .then(data => {
            if (!data || !data.due_reminders || !data.due_reminders.length) return;
            const items = data.due_reminders;
            if (banner) {
                banner.innerHTML = `
                    <div class="alert alert-warning border-0 shadow-sm d-flex align-items-start gap-3 mt-2" role="alert">
                        <i class="fas fa-bell fa-lg mt-1"></i>
                        <div class="flex-grow-1">
                            <strong>Deadline reminder${items.length > 1 ? 's' : ''}</strong>
                            <ul class="mb-1 mt-1 ps-3">
                                ${items.slice(0, 3).map(d => `<li>${escapeHtml(d.title)} - ${
                                    d.days_left < 0 ? Math.abs(d.days_left) + ' day(s) overdue' :
                                    d.days_left === 0 ? 'due today' : 'due in ' + d.days_left + ' day(s)'
                                } (${escapeHtml(d.due_date)})</li>`).join('')}
                            </ul>
                            <a href="/deadlines" class="small fw-semibold">View all deadlines</a>
                        </div>
                        <button type="button" class="btn-close" onclick="this.closest('.alert').remove()"></button>
                    </div>`;
            }
            // Browser notification (only if the user already allowed it) - once per day per deadline
            if ('Notification' in window && Notification.permission === 'granted') {
                const today = new Date().toISOString().slice(0, 10);
                items.forEach(d => {
                    const key = `le_notified_${d.id}_${today}`;
                    if (!localStorage.getItem(key)) {
                        localStorage.setItem(key, '1');
                        new Notification('LegalEase deadline: ' + d.title,
                            {body: `${d.due_date} - ${d.days_left < 0 ? 'overdue' : d.days_left + ' day(s) left'}`});
                    }
                });
            }
        })
        .catch(() => {});
});
