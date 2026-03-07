import { PrismaClient } from '@prisma/client'
import bcryptjs from 'bcryptjs'

const prisma = new PrismaClient()

async function main() {
  console.log('Starting seed...')

  // Clear existing data
  await prisma.kpiValue.deleteMany({})
  await prisma.weeklyRecord.deleteMany({})
  await prisma.uploadLog.deleteMany({})
  await prisma.operator.deleteMany({})
  await prisma.coordinator.deleteMany({})
  await prisma.user.deleteMany({})
  await prisma.machine.deleteMany({})
  await prisma.kpiDefinition.deleteMany({})

  // Create KPI Definitions
  const kpiDefinitions = [
    { code: 'BOS', name: 'BOS', category: 'S', categoryName: 'Sustentabilidad', unit: '#', sortOrder: 1 },
    { code: 'BOS_ENG', name: 'BOS ENG', category: 'S', categoryName: 'Sustentabilidad', unit: '%', sortOrder: 2 },
    { code: 'QBOS', name: 'QBOS', category: 'Q', categoryName: 'Calidad', unit: '#', sortOrder: 3 },
    { code: 'QBOS_ENG', name: 'QBOS ENG', category: 'Q', categoryName: 'Calidad', unit: '%', sortOrder: 4 },
    { code: 'QFLAGS', name: 'QFlags', category: 'Q', categoryName: 'Calidad', unit: '#', sortOrder: 5 },
    { code: 'QI_PNC', name: 'QI/PNC', category: 'Q', categoryName: 'Calidad', unit: '#', sortOrder: 6 },
    { code: 'DH_ENCONTRADOS', name: 'DH encontrados', category: 'D', categoryName: 'Defectos', unit: '#', sortOrder: 7 },
    { code: 'DH_REPARADOS', name: 'DH Reparados', category: 'D', categoryName: 'Defectos', unit: '#', sortOrder: 8 },
    { code: 'CURVA_AUTONOMIA', name: 'Curva de Autonomía', category: 'D', categoryName: 'Defectos', unit: '%', sortOrder: 9 },
    { code: 'CONTRAMEDIDAS', name: 'Contramedidas Defectos', category: 'D', categoryName: 'Defectos', unit: '%', sortOrder: 10 },
    { code: 'IPS', name: 'IPS', category: 'C', categoryName: 'Capacidad', unit: '#', sortOrder: 11 },
    { code: 'FRR', name: 'FRR', category: 'C', categoryName: 'Capacidad', unit: '%', sortOrder: 12 },
    { code: 'DIM_WASTE', name: 'DIM WASTE', category: 'C', categoryName: 'Capacidad', unit: '%', sortOrder: 13 },
    { code: 'SOBREPESO', name: 'Sobrepeso', category: 'C', categoryName: 'Capacidad', unit: '#', sortOrder: 14 },
    { code: 'EVENTOS_LAIKA', name: 'Eventos LAIKA', category: 'C', categoryName: 'Capacidad', unit: '#', sortOrder: 15 },
    { code: 'CASOS_ESTUDIOS', name: 'Casos de estudios', category: 'M', categoryName: 'Mejora', unit: '#', sortOrder: 16 },
    { code: 'QM_ON_TARGET', name: 'QM On Target', category: 'M', categoryName: 'Mejora', unit: '%', sortOrder: 17 },
  ]

  for (const kpi of kpiDefinitions) {
    await prisma.kpiDefinition.create({ data: kpi })
  }
  console.log('Created KPI definitions')

  // Create Machine
  const machine = await prisma.machine.create({
    data: {
      name: 'Filtros Blancos 1',
      code: 'FB1',
    },
  })
  console.log('Created machine:', machine.name)

  // Create Admin User
  const hashedAdminPassword = await bcryptjs.hash('admin123', 10)
  const adminUser = await prisma.user.create({
    data: {
      email: 'admin@fto.com',
      name: 'Admin User',
      passwordHash: hashedAdminPassword,
      role: 'ADMIN',
    },
  })
  console.log('Created admin user:', adminUser.email)

  // Coordinator and Operator data
  const coordinatorData = [
    {
      name: 'R Vargas',
      operators: [
        'Cristian Quintanilla',
        'Diana Magaña',
        'Luis Ramírez',
        'Miriam Ramírez',
        'Erick Macias',
        'Gilberto Magaña',
      ],
    },
    {
      name: 'C Lara',
      operators: [
        'Alexis Cruz',
        'Ramiro Mendez',
        'Victoria Karman',
        'Manuel Martinez',
        'Gilberto Alvarez',
      ],
    },
    {
      name: 'M Vidrio',
      operators: [
        'Jonathan Mares',
        'Roberto Mora',
        'Violeta Navarro',
        'Luis Vazquez',
        'Felipe García',
      ],
    },
  ]

  // Week dates for seed data
  const weekDates = [
    new Date('2026-01-19'),
    new Date('2026-01-26'),
    new Date('2026-02-02'),
    new Date('2026-02-09'),
    new Date('2026-02-16'),
    new Date('2026-02-23'),
  ]

  // Sample KPI values for each week and operator
  const sampleKpiValues = [
    { kpiCode: 'BOS', value: 95.5, objective: 98 },
    { kpiCode: 'BOS_ENG', value: 87.3, objective: 90 },
    { kpiCode: 'QBOS', value: 12, objective: 10 },
    { kpiCode: 'QBOS_ENG', value: 75.2, objective: 80 },
    { kpiCode: 'QFLAGS', value: 8, objective: 5 },
    { kpiCode: 'QI_PNC', value: 4, objective: 3 },
    { kpiCode: 'DH_ENCONTRADOS', value: 6, objective: 8 },
    { kpiCode: 'DH_REPARADOS', value: 5, objective: 6 },
    { kpiCode: 'CURVA_AUTONOMIA', value: 82.5, objective: 85 },
    { kpiCode: 'CONTRAMEDIDAS', value: 70.0, objective: 75 },
    { kpiCode: 'IPS', value: 145, objective: 150 },
    { kpiCode: 'FRR', value: 65.8, objective: 70 },
    { kpiCode: 'DIM_WASTE', value: 2.3, objective: 2.0 },
    { kpiCode: 'SOBREPESO', value: 3, objective: 2 },
    { kpiCode: 'EVENTOS_LAIKA', value: 1, objective: 0 },
    { kpiCode: 'CASOS_ESTUDIOS', value: 2, objective: 3 },
    { kpiCode: 'QM_ON_TARGET', value: 88.5, objective: 90 },
  ]

  // Create coordinators, operators, and weekly records
  for (const coordData of coordinatorData) {
    // Create coordinator user
    const hashedPassword = await bcryptjs.hash('password123', 10)
    const coordUser = await prisma.user.create({
      data: {
        email: `${coordData.name.toLowerCase().replace(/\s/g, '.')}@fto.com`,
        name: coordData.name,
        passwordHash: hashedPassword,
        role: 'COORDINATOR',
      },
    })

    // Create coordinator
    const coordinator = await prisma.coordinator.create({
      data: {
        name: coordData.name,
        machineId: machine.id,
        userId: coordUser.id,
      },
    })
    console.log('Created coordinator:', coordinator.name)

    // Create operators and weekly records
    for (const operatorName of coordData.operators) {
      const operator = await prisma.operator.create({
        data: {
          name: operatorName,
          coordinatorId: coordinator.id,
          machineId: machine.id,
        },
      })
      console.log('  Created operator:', operatorName)

      // Create weekly records with KPI values
      for (const weekStart of weekDates) {
        const weekNumber = getWeekNumber(weekStart)
        const year = weekStart.getFullYear()

        const weeklyRecord = await prisma.weeklyRecord.create({
          data: {
            operatorId: operator.id,
            weekStart,
            year,
            weekNumber,
            createdBy: adminUser.id,
          },
        })

        // Create KPI values for this week
        for (const kpiData of sampleKpiValues) {
          // Vary the values slightly for different weeks and operators
          const randomVariance = (Math.random() - 0.5) * 5
          await prisma.kpiValue.create({
            data: {
              weeklyRecordId: weeklyRecord.id,
              kpiCode: kpiData.kpiCode,
              value: Math.max(0, kpiData.value + randomVariance),
              objective: kpiData.objective,
            },
          })
        }
      }
    }
  }

  console.log('Seed completed successfully!')
}

function getWeekNumber(date: Date): number {
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()))
  const dayNum = d.getUTCDay() || 7
  d.setUTCDate(d.getUTCDate() + 4 - dayNum)
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1))
  return Math.ceil(((d.getTime() - yearStart.getTime()) / 86400000 + 1) / 7)
}

main()
  .then(async () => {
    await prisma.$disconnect()
  })
  .catch(async (e) => {
    console.error(e)
    await prisma.$disconnect()
    process.exit(1)
  })
