import { NextRequest, NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'

interface GetRecordsParams {
  machineId?: string
  coordinatorId?: string
  operatorId?: string
  weekStart?: string
  weekEnd?: string
  year?: string
  page?: string
  limit?: string
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const machineId = searchParams.get('machineId')
    const coordinatorId = searchParams.get('coordinatorId')
    const operatorId = searchParams.get('operatorId')
    const weekStart = searchParams.get('weekStart')
    const weekEnd = searchParams.get('weekEnd')
    const year = searchParams.get('year')
    const page = parseInt(searchParams.get('page') || '1')
    const limit = parseInt(searchParams.get('limit') || '50')

    const skip = (page - 1) * limit

    const where: any = {}

    if (operatorId) {
      where.operatorId = operatorId
    } else if (coordinatorId) {
      where.operator = {
        coordinatorId: coordinatorId
      }
    } else if (machineId) {
      where.operator = {
        machineId: machineId
      }
    }

    if (weekStart) {
      where.weekStart = {
        ...where.weekStart,
        gte: new Date(weekStart)
      }
    }

    if (weekEnd) {
      where.weekStart = {
        ...where.weekStart,
        lte: new Date(weekEnd)
      }
    }

    if (year) {
      where.year = parseInt(year)
    }

    const [records, total] = await Promise.all([
      prisma.weeklyRecord.findMany({
        where,
        include: {
          operator: {
            include: {
              coordinator: true
            }
          },
          kpiValues: true
        },
        orderBy: {
          weekStart: 'desc'
        },
        skip,
        take: limit
      }),
      prisma.weeklyRecord.count({ where })
    ])

    const formattedRecords = records.map((record) => ({
      id: record.id,
      operatorId: record.operatorId,
      operatorName: record.operator.name,
      coordinatorId: record.operator.coordinatorId,
      coordinatorName: record.operator.coordinator.name,
      weekStart: record.weekStart,
      year: record.year,
      weekNumber: record.weekNumber,
      values: Object.fromEntries(
        record.kpiValues.map((kv) => [kv.kpiCode, kv.value])
      )
    }))

    return NextResponse.json({
      records: formattedRecords,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit)
      }
    })
  } catch (error) {
    console.error('Error fetching records:', error)
    return NextResponse.json(
      { error: 'Failed to fetch records' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions)

    if (!session) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    const body = await request.json()
    const { operatorId, weekStart, kpiValues } = body

    if (!operatorId || !weekStart || !kpiValues) {
      return NextResponse.json(
        { error: 'operatorId, weekStart, and kpiValues are required' },
        { status: 400 }
      )
    }

    const weekStartDate = new Date(weekStart)
    const year = weekStartDate.getFullYear()
    const weekNumber = Math.ceil(
      (weekStartDate.getTime() - new Date(year, 0, 1).getTime()) / (7 * 24 * 60 * 60 * 1000)
    )

    // Validate that operator exists
    const operator = await prisma.operator.findUnique({
      where: { id: operatorId }
    })

    if (!operator) {
      return NextResponse.json(
        { error: 'Operator not found' },
        { status: 404 }
      )
    }

    // Validate that all KPI codes exist
    const kpiCodes = Object.keys(kpiValues)
    const validKpis = await prisma.kpiDefinition.findMany({
      where: {
        code: {
          in: kpiCodes
        }
      }
    })

    const validKpiCodes = validKpis.map((k) => k.code)
    const invalidCodes = kpiCodes.filter((code) => !validKpiCodes.includes(code))

    if (invalidCodes.length > 0) {
      return NextResponse.json(
        { error: `Invalid KPI codes: ${invalidCodes.join(', ')}` },
        { status: 400 }
      )
    }

    // Check if record already exists
    const existingRecord = await prisma.weeklyRecord.findUnique({
      where: {
        operatorId_weekStart: {
          operatorId,
          weekStart: weekStartDate
        }
      }
    })

    let record

    if (existingRecord) {
      // Update existing record
      await prisma.kpiValue.deleteMany({
        where: {
          weeklyRecordId: existingRecord.id
        }
      })

      record = await prisma.weeklyRecord.update({
        where: { id: existingRecord.id },
        data: {
          year,
          weekNumber,
          createdBy: session.user.id,
          kpiValues: {
            create: Object.entries(kpiValues).map(([code, value]) => ({
              kpiCode: code,
              value: value as number
            }))
          }
        },
        include: {
          kpiValues: true
        }
      })
    } else {
      // Create new record
      record = await prisma.weeklyRecord.create({
        data: {
          operatorId,
          weekStart: weekStartDate,
          year,
          weekNumber,
          createdBy: session.user.id,
          kpiValues: {
            create: Object.entries(kpiValues).map(([code, value]) => ({
              kpiCode: code,
              value: value as number
            }))
          }
        },
        include: {
          kpiValues: true
        }
      })
    }

    return NextResponse.json(
      {
        ...record,
        values: Object.fromEntries(
          record.kpiValues.map((kv) => [kv.kpiCode, kv.value])
        )
      },
      { status: existingRecord ? 200 : 201 }
    )
  } catch (error) {
    console.error('Error creating/updating record:', error)
    return NextResponse.json(
      { error: 'Failed to create or update record' },
      { status: 500 }
    )
  }
}
