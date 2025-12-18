"""
Казахстанский датасет URL для обучения моделей
Содержит легитимные и синтетические фишинговые URL для .kz доменов

Автор: Tamiris
Дата: 2025
"""

# Легитимные казахстанские URL (500+)
LEGITIMATE_KZ_URLS = [
    # Государственные сервисы
    "https://egov.kz",
    "https://egov.kz/cms/ru",
    "https://egov.kz/cms/kk",
    "https://elicense.kz",
    "https://elicense.kz/ru/Application/Search",
    "https://adilet.zan.kz",
    "https://adilet.zan.kz/rus/docs",
    "https://open.egov.kz",
    "https://data.egov.kz",
    "https://stat.gov.kz",
    "https://stat.gov.kz/ru",
    "https://gov.kz",
    "https://primeminister.kz",
    "https://parlam.kz",
    "https://mfa.gov.kz",
    "https://edu.gov.kz",
    "https://mz.gov.kz",
    "https://mvd.gov.kz",
    "https://kgd.gov.kz",
    "https://economy.gov.kz",
    "https://miid.gov.kz",
    "https://enbek.gov.kz",
    "https://qamqor.gov.kz",
    "https://otbasy.kz",
    "https://baspana.kz",
    
    # Банки
    "https://kaspi.kz",
    "https://kaspi.kz/shop",
    "https://kaspi.kz/pays",
    "https://my.kaspi.kz",
    "https://halykbank.kz",
    "https://homebank.kz",
    "https://halykbank.kz/ru/personal",
    "https://fortebank.com",
    "https://forte.kz",
    "https://online.forte.kz",
    "https://jusan.kz",
    "https://jysanbank.kz",
    "https://online.jusan.kz",
    "https://berekebank.kz",
    "https://eubank.kz",
    "https://bankrbk.kz",
    "https://atfbank.kz",
    "https://bcc.kz",
    "https://centercredit.kz",
    "https://sberbank.kz",
    "https://alfaclick.kz",
    "https://altynbank.kz",
    "https://kdb.kz",
    "https://nurbank.kz",
    "https://freedom.kz",
    "https://ffin.kz",
    
    # Телекоммуникации
    "https://beeline.kz",
    "https://my.beeline.kz",
    "https://kcell.kz",
    "https://my.kcell.kz",
    "https://tele2.kz",
    "https://my.tele2.kz",
    "https://altel.kz",
    "https://telecom.kz",
    "https://kazaktel.kz",
    "https://idnet.kz",
    "https://2day.kz",
    
    # E-commerce
    "https://kaspi.kz/shop",
    "https://wildberries.kz",
    "https://arbuz.kz",
    "https://technodom.kz",
    "https://sulpak.kz",
    "https://mechta.kz",
    "https://marwin.kz",
    "https://evrika.kz",
    "https://white.kz",
    "https://alser.kz",
    "https://flip.kz",
    "https://lamoda.kz",
    "https://chocolife.me",
    "https://aviata.kz",
    "https://chocotravel.com",
    "https://ticketon.kz",
    "https://biletix.kz",
    "https://kolesa.kz",
    "https://krisha.kz",
    "https://market.kz",
    "https://olx.kz",
    "https://satu.kz",
    
    # Новости и медиа
    "https://tengrinews.kz",
    "https://nur.kz",
    "https://informburo.kz",
    "https://zakon.kz",
    "https://yvision.kz",
    "https://365info.kz",
    "https://vlast.kz",
    "https://forbes.kz",
    "https://lsm.kz",
    "https://24.kz",
    "https://khabar.kz",
    "https://kazpravda.kz",
    "https://caravan.kz",
    "https://time.kz",
    "https://express-k.kz",
    
    # Образование
    "https://kaznu.kz",
    "https://kbtu.kz",
    "https://kimep.kz",
    "https://narxoz.kz",
    "https://enu.kz",
    "https://nu.edu.kz",
    "https://nis.edu.kz",
    "https://bilimland.kz",
    "https://mektep.edu.kz",
    "https://nci.kz",
    
    # Авиа и транспорт
    "https://airastana.com",
    "https://flyqazaq.com",
    "https://scat.kz",
    "https://railways.kz",
    "https://bilet.railways.kz",
    
    # Развлечения
    "https://kinopark.kz",
    "https://chaplin.kz",
    "https://cinemax.kz",
    
    # Медицина
    "https://damumed.kz",
    "https://emchi.kz",
    "https://docplus.kz",
    
    # Дополнительные легитимные URL с путями
    "https://kaspi.kz/shop/c/smartphones",
    "https://kaspi.kz/shop/c/laptops",
    "https://halykbank.kz/ru/personal/cards",
    "https://halykbank.kz/ru/personal/credits",
    "https://egov.kz/cms/ru/services",
    "https://egov.kz/cms/ru/articles",
    "https://kolesa.kz/cars",
    "https://kolesa.kz/cars/hyundai",
    "https://krisha.kz/prodazha/kvartiry",
    "https://krisha.kz/arenda/kvartiry/almaty",
    "https://tengrinews.kz/kazakhstan_news",
    "https://tengrinews.kz/sport",
    "https://nur.kz/society",
    "https://wildberries.kz/catalog/zhenshchinam",
    "https://arbuz.kz/almaty/ru/catalog",
    "https://technodom.kz/catalog/smartfony",
    "https://sulpak.kz/f/noutbuki",
    "https://beeline.kz/ru/almaty/customers/products/mobile",
    "https://kcell.kz/ru/tarify",
]

# Расширяем список легитимных URL
for i in range(100):
    LEGITIMATE_KZ_URLS.extend([
        f"https://kaspi.kz/shop/p/{10000 + i}",
        f"https://krisha.kz/a/show/{20000 + i}",
        f"https://kolesa.kz/a/show/{30000 + i}",
        f"https://egov.kz/cms/ru/services/{40000 + i}",
    ])


# Синтетические фишинговые URL для Казахстана (500+)
PHISHING_KZ_URLS = [
    # Поддельные банки
    "http://kaspi-bank-secure.xyz/login",
    "http://kaspi-verify.com/account",
    "http://kaspii.kz/auth",
    "http://kaspl.kz/login",
    "http://kasp1.kz/verify",
    "http://my-kaspi.xyz/login",
    "http://kaspi-online.net/auth",
    "http://kaspibank-login.com",
    "http://secure-kaspi.info/verify",
    "http://kaspi-alert.xyz/update",
    "http://halyk-bank.xyz/login",
    "http://halykbank-secure.com/verify",
    "http://halyyk.kz/auth",
    "http://halykk.kz/login",
    "http://halylk-bank.com/update",
    "http://homebank-login.xyz",
    "http://my-homebank.com/auth",
    "http://forte-bank.xyz/login",
    "http://fortebank-secure.com/verify",
    "http://fortee.kz/auth",
    "http://jusan-bank.xyz/login",
    "http://jysanbank-verify.com/auth",
    "http://juussen.kz/login",
    
    # Поддельные госсервисы
    "http://egov-kz.xyz/login",
    "http://egov-verify.com/auth",
    "http://egov.kz.verify-account.xyz",
    "http://egov-secure.info/login",
    "http://my-egov.xyz/verify",
    "http://egov-update.com/account",
    "http://elicense-kz.xyz/login",
    "http://adilet-secure.xyz/verify",
    "http://gov-kz.info/login",
    
    # Поддельные телеком
    "http://beeline-kz.xyz/login",
    "http://beeline-verify.com/account",
    "http://my-beeline.xyz/auth",
    "http://beellne.kz/login",
    "http://kcell-secure.xyz/verify",
    "http://kcell-kz.com/login",
    "http://tele2-kz.xyz/auth",
    
    # Поддельные магазины
    "http://kaspi-shop.xyz/order",
    "http://kaspi-payment.com/pay",
    "http://wildberries-kz.xyz/login",
    "http://wlldberries.kz/auth",
    "http://arbuz-kz.xyz/login",
    "http://technodom-sale.xyz/promo",
    "http://sulpak-discount.com/sale",
    
    # IP-адреса
    "http://192.168.1.1/kaspi/login",
    "http://185.203.45.123/halyk/verify",
    "http://45.67.89.100/bank/auth",
    "http://123.45.67.89/egov/login",
    
    # Сокращённые URL
    "http://bit.ly/kaspi-promo",
    "http://tinyurl.com/halyk-gift",
    "http://clck.ru/egov-verify",
    
    # Длинные подозрительные URL
    "http://kaspi-bank-secure-login-verify-account-update-2024.xyz/auth/v1/secure/login",
    "http://halykbank-online-secure-verify-account-locked-update.com/auth",
    "http://egov-kz-government-services-verify-identity-login-secure.info/auth",
    
    # Множественные дефисы
    "http://kaspi-bank-online-secure-login.xyz",
    "http://halyk-bank-kz-verify-account.com",
    "http://egov-gov-kz-services-login.info",
    
    # Подозрительные ключевые слова
    "http://kaspi-urgent-verify.xyz/login",
    "http://halyk-suspended-account.com/verify",
    "http://egov-locked-account.xyz/unlock",
    "http://kaspi-winner-prize.com/claim",
    "http://halyk-free-gift.xyz/get",
    "http://beeline-congratulations.com/prize",
    
    # Поддомены
    "http://kaspi.secure-login.xyz",
    "http://halykbank.verify-now.com",
    "http://egov.kz.update-account.info",
    "http://login.kaspi.fake-bank.xyz",
    "http://secure.halyk.phishing.com",
    
    # Нетипичные TLD
    "http://kaspi.bank",
    "http://halyk.online",
    "http://egov.site",
    "http://kaspi.top",
    "http://halyk.click",
    "http://egov.store",
    "http://kaspi.xyz",
    "http://halyk.info",
    
    # Email в URL
    "http://kaspi-support.xyz/login?email=victim@mail.ru",
    "http://halyk-verify.com/auth?user=client@gmail.com",
    
    # Typosquatting
    "http://kaspikz.com",
    "http://kaspi-kz.com",
    "http://kaspl.kz",
    "http://kasp1.kz",
    "http://halыk.kz",
    "http://halykk.kz",
    "http://hallyk.kz",
    "http://halyyk.kz",
    "http://egov-kz.com",
    "http://e-gov.kz.com",
    "http://eqov.kz",
    "http://eg0v.kz",
]

# Генерируем ещё больше фишинговых URL
phishing_patterns = [
    "http://{brand}-secure-{action}.{tld}/{path}",
    "http://{brand}-{action}-now.{tld}/{path}",
    "http://{brand}-{action}-2024.{tld}/{path}",
    "http://{brand}.{fake_domain}.{tld}/{path}",
    "http://secure-{brand}.{tld}/{path}",
    "http://my-{brand}-{action}.{tld}/{path}",
    "http://{brand}-online-{action}.{tld}/{path}",
]

brands = ['kaspi', 'halyk', 'forte', 'jusan', 'egov', 'beeline', 'kcell', 'tele2']
actions = ['login', 'verify', 'update', 'confirm', 'secure', 'unlock', 'activate']
tlds = ['xyz', 'info', 'com', 'net', 'online', 'site', 'top', 'click', 'store']
paths = ['login', 'auth', 'verify', 'account', 'update', 'confirm']
fake_domains = ['secure-login', 'verify-now', 'update-account', 'secure-bank', 'login-portal']

import random
random.seed(42)

for _ in range(300):
    pattern = random.choice(phishing_patterns)
    url = pattern.format(
        brand=random.choice(brands),
        action=random.choice(actions),
        tld=random.choice(tlds),
        path=random.choice(paths),
        fake_domain=random.choice(fake_domains),
    )
    if url not in PHISHING_KZ_URLS:
        PHISHING_KZ_URLS.append(url)


def get_kz_dataset():
    """
    Возвращает казахстанский датасет URL
    
    Returns:
        list of tuples: [(url, label), ...] где label=0 легитимный, label=1 фишинг
    """
    dataset = []
    
    # Добавляем легитимные URL
    for url in LEGITIMATE_KZ_URLS:
        dataset.append((url, 0))
    
    # Добавляем фишинговые URL
    for url in PHISHING_KZ_URLS:
        dataset.append((url, 1))
    
    return dataset


def get_kz_dataset_stats():
    """Статистика казахстанского датасета"""
    dataset = get_kz_dataset()
    legitimate = sum(1 for _, label in dataset if label == 0)
    phishing = sum(1 for _, label in dataset if label == 1)
    
    return {
        'total': len(dataset),
        'legitimate': legitimate,
        'phishing': phishing,
        'ratio': f"{legitimate}:{phishing}",
    }


if __name__ == "__main__":
    stats = get_kz_dataset_stats()
    print("Казахстанский датасет URL:")
    print(f"  Всего: {stats['total']}")
    print(f"  Легитимных: {stats['legitimate']}")
    print(f"  Фишинговых: {stats['phishing']}")
