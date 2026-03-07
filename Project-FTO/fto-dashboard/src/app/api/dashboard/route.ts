import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'

interface ObjectiveData {
  [kpiCode: string]: number
}

interface WeeklyData {
  week: string
  values: Record<string, number>
}

interface FormattedOperator {
  id: string
  name: string
  objectives: ObjectiveData
  weeklyData: WeeklyData[]
}

interface FormattedCoordinator {
  id: string
  name: string
  operators: FormattedOperator[]
}

interface DashboardDataResponse {
  machine: {
    id: string
    name: string
    code: string
  }
  coordinators: FormattedCoordinator[]
  kpiDefinitions: any[]
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const machineId = searchParams.get('machineId')

    if (!machineId) {
      return NextResponse.json(
        { error: 'machineId query parameter is required' },
        { status: 400 }
      )
    }

    // Fetch machine
    const machine = await prisma.machine.findUnique({
      where: { id: machineId }
    })

    if (!machine) {
      return NextResponse.json(
        { error: 'Machine not found' },
        { status: 404 }
      )
    }

    // Fetch coordinators and operators for this machine
    const coordinators = await prisma.coordinator.findMany({
      where: { machineId },
      include: {
        operators: {
          include: {
            weeklyRecords: {
              include: {
                kpiValues: true
              },
              orderBy: {
                weekStart: 'asc'
              }
            }
          }
        }
      }
    })

    // Fetch KPI definitions
    const kpiDefinitions = await prisma.kpiDefinition.findMany({
      orderBy: {
        sortOrder: 'asc'
      }
    })

    // Format the response
    const formattedCoordinators: FormattedCoordinator[] = coordinators.map((coordinator) => ({
      id: coordinator.id,
      name: coordinator.name,
      operators: coordinator.operators.map((operator) => {
        // Calculate objectives from first available weekly record or use defaults
        const objectives: ObjectiveData = {}
        kpiDefinitions.forEach((kpi) => {
          objectives[kpi.code] = kpi.defaultObjective || 0
        })

        // Collect all objectives from weekly records
        operator.weeklyRecords.forEach((record) => {
          record.kpiValues.forEach((kpiValue) => {
            if (kpiValue.objective !== null && kpiValue.objective !== undefined) {
              objectives[kpiValue.kpiCode] = kpiValue.objective
            }
          })
        })

        const weeklyData: WeeklyData[] = operator.weeklyRecords.map((record) => ({
          week: record.weekStart.toISOString().split('T')[0],
          values: Object.fromEntries(
            record.kpiValues.map((kv) => [kv.kpiCode, kv.value])
          )
        }))

        return {
          id: operator.id,
          name: operator.name,
          objectives,
          weeklyData
        }
      })
    }))

    const response: DashboardDataResponse = {
      machine: {
        id: machine.id,
        name: machine.name,
        code: machine.code
      },
      coordinators: formattedCoordinators,
      kpiDefinitions
    }

    return NextResponse.json(response)
  } catch (error) {
    console.error('Error fetching dashboard data:', error)
    return NextResponse.json(
      { error: 'Failed to fetch dashboard data' },
      { status: 500 }
    )
  }
}
