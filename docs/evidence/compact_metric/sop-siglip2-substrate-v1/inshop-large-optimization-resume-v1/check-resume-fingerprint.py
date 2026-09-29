import pe_large_optimization as driver
assert driver.fingerprint([1,23])!=driver.fingerprint([12,3]), 'resume fingerprint loses scalar boundaries'
print('PASS scalar boundaries are preserved')
